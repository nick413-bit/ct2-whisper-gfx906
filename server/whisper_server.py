import argparse,logging,os,tempfile,time,subprocess,json,asyncio
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional,List
import numpy as np
rocr=os.environ.get("ROCR_VISIBLE_DEVICES","")
if rocr:
    os.environ["HIP_VISIBLE_DEVICES"]=rocr
import uvicorn
from fastapi import FastAPI,File,Form,HTTPException,UploadFile,Request
from fastapi.responses import JSONResponse,PlainTextResponse,HTMLResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log=logging.getLogger("whisper")
MODELS=[];DIARIZE=None;MNAME="whisper";MPATH=""
BEAM_SIZE=2;NUM_GPUS=1
EVENT_QUEUES={}
def preprocess_audio(audio_path, sr=16000):
    import soundfile as sf
    import noisereduce as nr
    from scipy.signal import butter, sosfilt
    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False);tmp.close()
    subprocess.run(["ffmpeg","-i",audio_path,"-ar",str(sr),"-ac","1","-f","wav","-y",tmp.name],capture_output=True)
    y,_=sf.read(tmp.name,dtype="float32");os.unlink(tmp.name)
    log.info("Preprocess: loaded %.1fs at %dHz",len(y)/sr,sr)
    y=nr.reduce_noise(y=y,sr=sr,prop_decrease=0.6,stationary=False)
    sos=butter(5,[80,7600],btype='bandpass',fs=sr,output='sos')
    y=sosfilt(sos,y).astype(np.float32)
    rms=np.sqrt(np.mean(y**2))
    if rms>0: y=y*(0.1/rms)
    y=np.clip(y,-1.0,1.0)
    import soundfile as sf2
    out=tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
    sf2.write(out.name,y,sr);log.info("Preprocess: done")
    return out.name
def make_wav_16k(src):
    out=tempfile.NamedTemporaryFile(suffix=".wav",delete=False);out.close()
    subprocess.run(["ffmpeg","-i",src,"-ar","16000","-ac","1","-f","wav","-y",out.name],capture_output=True)
    return out.name
def find_split_points(wav_path, n_chunks, sr=16000):
    import soundfile as sf
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    y,_=sf.read(wav_path,dtype="float32")
    total=len(y)
    if n_chunks<=1:
        return [(0,total)]
    opts=VadOptions(min_silence_duration_ms=300)
    speech=get_speech_timestamps(y,vad_options=opts)
    pauses=[]
    for i in range(len(speech)-1):
        pause_start=speech[i]["end"]
        pause_end=speech[i+1]["start"]
        mid=(pause_start+pause_end)//2
        pauses.append(mid)
    log.info("VAD found %d pauses for splitting",len(pauses))
    chunk_sz=total//n_chunks
    split_pts=[0]
    for i in range(1,n_chunks):
        ideal=i*chunk_sz
        best=ideal
        best_dist=total
        for p in pauses:
            dist=abs(p-ideal)
            if dist<best_dist:
                best_dist=dist;best=p
        split_pts.append(best)
    split_pts.append(total)
    ranges=[]
    for i in range(len(split_pts)-1):
        ranges.append((split_pts[i],split_pts[i+1]))
    return ranges
def split_audio(wav_path, n_chunks):
    import soundfile as sf
    y,sr=sf.read(wav_path,dtype="float32")
    total=len(y)
    if n_chunks<=1:
        return [(wav_path,0.0)]
    ranges=find_split_points(wav_path,n_chunks,sr)
    chunks=[]
    for i,(s,e) in enumerate(ranges):
        f=tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
        sf.write(f.name,y[s:e],sr);f.close()
        offset=s/sr
        log.info("Chunk %d: %.1fs-%.1fs (%.1fs)",i,s/sr,e/sr,(e-s)/sr)
        chunks.append((f.name,offset))
    return chunks
def transcribe_chunk(gpu_idx, chunk_path, offset, language=None, prompt=None, temperature=0.0):
    model=MODELS[gpu_idx]
    kw={}
    if language: kw["language"]=language
    if prompt: kw["initial_prompt"]=prompt
    if temperature and temperature>0: kw["temperature"]=temperature
    segs,info=model.transcribe(chunk_path,beam_size=BEAM_SIZE,vad_filter=True,vad_parameters=dict(min_silence_duration_ms=500),**kw)
    results=[]
    for s in segs:
        results.append({"start":round(s.start+offset,3),"end":round(s.end+offset,3),"text":s.text.strip()})
    log.info("GPU %d: transcribed %.1fs chunk -> %d segments",gpu_idx,offset,len(results))
    return results
def parallel_transcribe(wav_path, language=None, prompt=None, temperature=0.0):
    n=min(NUM_GPUS,len(MODELS))
    if n<=1:
        return transcribe_chunk(0,wav_path,0.0,language,prompt,temperature)
    chunks=split_audio(wav_path,n)
    log.info("Parallel transcription on %d GPUs",n)
    t0=time.time()
    all_segs=[None]*n
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs={}
        for i,(cp,off) in enumerate(chunks):
            futs[ex.submit(transcribe_chunk,i,cp,off,language,prompt,temperature)]=i
        for fut in as_completed(futs):
            idx=futs[fut]
            all_segs[idx]=fut.result()
    for cp,_ in chunks:
        try: os.unlink(cp)
        except: pass
    merged=[]
    for s in all_segs:
        if s: merged.extend(s)
    merged.sort(key=lambda x:x["start"])
    log.info("Parallel done in %.1fs, %d segments",time.time()-t0,len(merged))
    return merged
def run_diarization(wav_path, min_speakers=None, max_speakers=None, num_speakers=None):
    if DIARIZE is None: return None
    import torch
    log.info("Diarization: num=%s min=%s max=%s", num_speakers, min_speakers, max_speakers)
    t0=time.time()
    kwargs={}
    if num_speakers is not None and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers and min_speakers > 0:
            kwargs["min_speakers"] = min_speakers
        if max_speakers and max_speakers > 0:
            kwargs["max_speakers"] = max_speakers
    with torch.cuda.device(0):
        dia = DIARIZE(wav_path, **kwargs)
    n_spk = len(set(s for _,_,s in dia.itertracks(yield_label=True)))
    log.info("Diarization: %.1fs, %d speakers", time.time()-t0, n_spk)
    turns=[]
    for turn,_,spk in dia.itertracks(yield_label=True):
        turns.append({"start":turn.start,"end":turn.end,"speaker":spk})
    return turns
def assign_speakers(segments, diarization):
    if not diarization: return segments
    for seg in segments:
        best=None;best_ov=0
        for d in diarization:
            ov=max(0,min(seg["end"],d["end"])-max(seg["start"],d["start"]))
            if ov>best_ov: best_ov=ov;best=d["speaker"]
        seg["speaker"]=best or "UNKNOWN"
    return segments
def resolve_model_path(path):
    p=Path(path)
    if (p/"model.bin").exists(): return str(p)
    name=p.name
    m={"whisper-large-v3-turbo":"large-v3-turbo","whisper-large-v3":"large-v3","whisper-large-v2":"large-v2","whisper-medium":"medium","whisper-small":"small","whisper-tiny":"tiny","whisper-base":"base"}
    for k,v in m.items():
        if k in name: log.info("Using size=%s",v); return v
    return path
def load_models(path,dev="cuda",ct="int8",n_gpus=1):
    from faster_whisper import WhisperModel
    rp=resolve_model_path(path)
    models=[]
    for i in range(n_gpus):
        log.info("Loading whisper on GPU %d (%s ct=%s)",i,rp,ct)
        m=WhisperModel(rp,device=dev,device_index=i,compute_type=ct)
        models.append(m)
        log.info("GPU %d: whisper loaded",i)
    return models
def load_diarization(dev="cuda:0"):
    try:
        import torch
        from pyannote.audio import Pipeline
        token=os.environ.get("HF_TOKEN","")
        if not token:
            log.warning("No HF_TOKEN, diarization disabled")
            return None
        log.info("Loading pyannote diarization on %s...",dev)
        p=Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token=token)
        try:
            p.to(torch.device(dev))
            log.info("Diarization on %s",dev)
        except Exception as e:
            log.warning("Diarization GPU failed (%s), using CPU",e)
        return p
    except Exception as e:
        log.warning("Diarization load failed: %s",e)
        return None
@asynccontextmanager
async def lifespan(a):
    global MODELS,DIARIZE,NUM_GPUS
    import torch,ctranslate2
    torch.cuda.init()
    torch.cuda.set_device(0)
    NUM_GPUS=ctranslate2.get_cuda_device_count()
    log.info("Available GPUs: %d",NUM_GPUS)
    DIARIZE=load_diarization("cuda:0")
    ct=os.environ.get("WHISPER_COMPUTE_TYPE","int8")
    MODELS=load_models(MPATH,"cuda",ct,NUM_GPUS)
    log.info("All %d models loaded, beam=%d, ct=%s",len(MODELS),BEAM_SIZE,ct)
    yield
    MODELS=[];DIARIZE=None
app=FastAPI(title="CT2-Whisper ROCm Multi-GPU (gfx906)",lifespan=lifespan)
os.makedirs("/app/static",exist_ok=True)
app.mount("/static",StaticFiles(directory="/app/static"),name="static")
JOB_STATUS={"stage":"idle","message":"","progress":0,"logs":[]}
def emit_event(stage, message, progress=0):
    JOB_STATUS["stage"]=stage
    JOB_STATUS["message"]=message
    JOB_STATUS["progress"]=progress
    JOB_STATUS["logs"].append({"ts":time.time(),"msg":message})
    if len(JOB_STATUS["logs"])>50:
        JOB_STATUS["logs"]=JOB_STATUS["logs"][-50:]
@app.get("/",response_class=HTMLResponse)
async def gui():
    f=Path("/app/static/index.html")
    if f.exists(): return f.read_text()
    return "<h1>CT2-Whisper Multi-GPU</h1>"
@app.get("/v1/models")
async def list_models():
    return {"object":"list","data":[{"id":MNAME,"object":"model","created":int(time.time()),"owned_by":"local"}]}
@app.get("/v1/models/{mid}")
async def get_model(mid:str):
    return {"id":MNAME,"object":"model","created":int(time.time()),"owned_by":"local"}
@app.get("/health")
async def health():
    return {"status":"ok","models":len(MODELS),"gpus":NUM_GPUS,"diarization":DIARIZE is not None,"beam_size":BEAM_SIZE}
@app.get("/v1/status")
async def get_status():
    return JSONResponse(JOB_STATUS)
@app.post("/v1/audio/transcriptions")
async def transcribe(file:UploadFile=File(...),model:Optional[str]=Form(None),language:Optional[str]=Form(None),prompt:Optional[str]=Form(None),response_format:Optional[str]=Form("json"),temperature:Optional[float]=Form(0.0),diarize:Optional[str]=Form("auto"),preprocessing:Optional[str]=Form("auto"),num_speakers:Optional[int]=Form(None),min_speakers:Optional[int]=Form(None),max_speakers:Optional[int]=Form(None)):
    if not MODELS: raise HTTPException(503,"not loaded")
    JOB_STATUS["logs"]=[]
    emit_event("upload","Receiving file...",5)
    t_total=time.time()
    data=await file.read()
    sfx=Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=sfx,delete=True) as tmp:
        tmp.write(data);tmp.flush()
        emit_event("upload","Converting to WAV 16kHz...",8)
        wav_file=make_wav_16k(tmp.name)
    def _heavy():
        audio_file=wav_file;pp_file=None
        try:
            if preprocessing!="off":
                try:
                    emit_event("preprocess","Noise reduction & filtering...",15)
                    pp_file=preprocess_audio(wav_file)
                    audio_file=pp_file
                    emit_event("preprocess","Preprocessing done",20)
                except Exception as e:
                    log.warning("Preprocess err: %s",e)
            dia_turns=None
            if diarize!="off" and DIARIZE is not None:
                emit_event("diarize","Speaker diarization on GPU...",25)
                dia_turns=run_diarization(wav_file,min_speakers=min_speakers,max_speakers=max_speakers,num_speakers=num_speakers)
                ns=len(set(d["speaker"] for d in dia_turns)) if dia_turns else 0
                emit_event("diarize",f"Diarization done: {ns} speakers",40)
            ng=min(NUM_GPUS,len(MODELS))
            emit_event("transcribe",f"Transcribing on {ng} GPUs...",45)
            segments=parallel_transcribe(audio_file,language,prompt,temperature)
            emit_event("transcribe",f"{len(segments)} segments transcribed",85)
            if dia_turns:
                emit_event("transcribe","Assigning speakers...",90)
                segments=assign_speakers(segments,dia_turns)
            text=" ".join(s["text"] for s in segments)
            emit_event("done","Complete!",100)
            return segments,text,dia_turns
        finally:
            for ff in [wav_file,pp_file]:
                if ff:
                    try: os.unlink(ff)
                    except: pass
    segments,text,dia_turns=await run_in_threadpool(_heavy)

    elapsed=time.time()-t_total
    log.info("Total: %.1fs for %d segments",elapsed,len(segments))
    emit_event("done",f"Complete! {len(segments)} segments in {elapsed:.1f}s",100)
    if response_format=="verbose_json":
        return {"text":text,"segments":segments,"language":language,"processing_time":round(elapsed,1),"gpus_used":min(NUM_GPUS,len(MODELS))}
    if response_format=="text": return PlainTextResponse(text)
    if dia_turns:
        return {"text":text,"segments":segments,"processing_time":round(elapsed,1)}
    return {"text":text}
@app.post("/v1/audio/translations")
async def translate_audio(file:UploadFile=File(...),model:Optional[str]=Form(None),prompt:Optional[str]=Form(None),response_format:Optional[str]=Form("json"),temperature:Optional[float]=Form(0.0)):
    if not MODELS: raise HTTPException(503,"not loaded")
    data=await file.read()
    sfx=Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=sfx,delete=True) as tmp:
        tmp.write(data);tmp.flush()
        segs=parallel_transcribe(tmp.name,prompt=prompt,temperature=temperature)
        text=" ".join(s["text"] for s in segs)
    if response_format=="text": return PlainTextResponse(text)
    return {"text":text}
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model-path","--model_path",required=True)
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--host",default="0.0.0.0")
    p.add_argument("--model-name","--model_name","--served-model-name",default="whisper")
    p.add_argument("--device",default=None)
    p.add_argument("--compute-type","--compute_type",default=None)
    p.add_argument("--beam-size","--beam_size",type=int,default=None)
    p.add_argument("--num-gpus","--num_gpus",type=int,default=None)
    return p.parse_known_args()
if __name__=="__main__":
    args,unk=parse_args()
    MPATH=args.model_path;MNAME=args.model_name
    if args.device: os.environ["WHISPER_DEVICE"]=args.device
    if args.compute_type: os.environ["WHISPER_COMPUTE_TYPE"]=args.compute_type
    if args.beam_size: BEAM_SIZE=args.beam_size
    if args.num_gpus: NUM_GPUS=args.num_gpus
    if unk: log.warning("Ignoring unknown args: %s",unk)
    log.info("Starting: path=%s port=%d beam=%d",MPATH,args.port,BEAM_SIZE)
    uvicorn.run(app,host=args.host,port=args.port,log_level="info")
