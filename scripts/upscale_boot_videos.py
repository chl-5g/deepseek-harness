import os, subprocess, tempfile
from pathlib import Path
import cv2, torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'plugins/dsh-boot-animation/media'
MODEL = ROOT / '.models/RealESRGAN_x4plus.pth'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
ups = RealESRGANer(scale=4, model_path=str(MODEL), model=model, tile=512, tile_pad=16, pre_pad=0, half=False, device=device)

for src in sorted(MEDIA.glob('*.mp4')):
    probe = cv2.VideoCapture(str(src))
    if int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) >= 3840 and int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) >= 2160:
        probe.release()
        print(f'skip completed {src.name}', flush=True)
        continue
    probe.release()
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); tmp = Path(tempfile.mkdtemp(prefix='dsh-upscale-'))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        out, _ = ups.enhance(frame, outscale=3)
        cv2.imwrite(str(tmp / f'{i:06d}.png'), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        i += 1
        if i % 30 == 0: print(f'{src.name}: {i}/{frames}', flush=True)
    cap.release()
    stem = src.with_suffix('')
    silent = stem.with_name(stem.name + '-4k-silent.mp4')
    subprocess.run(['ffmpeg','-y','-framerate',str(fps),'-i',str(tmp/'%06d.png'),'-c:v','libx264','-crf','16','-preset','slow','-pix_fmt','yuv420p',str(silent)], check=True)
    final = stem.with_name(stem.name + '-4k.mp4')
    subprocess.run(['ffmpeg','-y','-i',str(silent),'-i',str(src),'-map','0:v:0','-map','1:a?','-c:v','copy','-c:a','copy','-shortest',str(final)], check=True)
    silent.unlink(missing_ok=True)
    src.unlink()
    final.rename(src)
    print(f'finished {src.name}', flush=True)
