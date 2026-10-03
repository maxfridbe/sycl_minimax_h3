//! Writing a clip: frames (and sound) -> .mp4, through ffmpeg (H.264 + AAC), the way the reference pipeline writes
//! its clips (reference/h3x.py `_write_video`): 24 fps, yuv420p, the sound cut or padded to exactly the video's
//! length and peak-normalized to -3 dBFS.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use h3_core::{Ctx, Error, Result};

/// Sound for a clip: one buffer per channel.
pub struct Audio {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

/// Pictures for a clip: [3, count, height, width] in [0, 1] (planar, as the decoder writes them).
pub struct Frames<'a> {
    pub px: &'a [f32],
    pub count: usize,
    pub height: usize,
    pub width: usize,
    pub fps: u32,
}

pub fn write_mp4(path: &Path, v: &Frames, audio: Option<&Audio>, log: &mut dyn FnMut(String)) -> Result<()> {
    let (frames, f, h, w, fps) = (v.px, v.count, v.height, v.width, v.fps);
    let plane = h * w;
    if frames.len() != 3 * f * plane {
        return Err(Error(format!("write_mp4: {} values for 3 x {f} x {h} x {w}", frames.len())));
    }
    let tmp = path.with_extension("pcm.tmp");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-"]);
    if let Some(a) = audio {
        // the length lock: exactly F / fps seconds of sound
        let want = ((f as f64 / fps as f64) * a.sample_rate as f64).round() as usize;
        let peak = a.channels.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
        let gain = if peak > 1e-6 { 10f32.powf(-3.0 / 20.0) / peak } else { 1.0 };
        log(format!("audio  : peak {:.1} dBFS -> -3 dBFS (gain x{gain:.1})", 20.0 * peak.max(1e-12).log10()));
        let ch = a.channels.len();
        let mut pcm = Vec::with_capacity(want * ch * 2);
        for i in 0..want {
            for c in &a.channels {
                let v = c.get(i).copied().unwrap_or(0.0) * gain;
                pcm.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
            }
        }
        std::fs::write(&tmp, pcm).ctx("writing the sound for ffmpeg")?;
        cmd.args(["-f", "s16le", "-ar", &a.sample_rate.to_string(), "-ac", &ch.to_string(), "-i"]).arg(&tmp);
        cmd.args(["-c:a", "aac"]);
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]).arg(path);
    let mut child = cmd.stdin(Stdio::piped()).spawn().ctx("starting ffmpeg")?;
    {
        let mut stdin = child.stdin.take().ok_or("ffmpeg has no stdin")?;
        let mut rgb = vec![0u8; plane * 3];
        for fi in 0..f {
            for c in 0..3 {
                let src = &frames[(c * f + fi) * plane..][..plane];
                for (i, v) in src.iter().enumerate() {
                    rgb[i * 3 + c] = (v.clamp(0.0, 1.0) * 255.0) as u8;
                }
            }
            stdin.write_all(&rgb).ctx("writing frames to ffmpeg")?;
        }
    }
    let st = child.wait()?;
    let _ = std::fs::remove_file(&tmp);
    if !st.success() {
        return Err(Error(format!("ffmpeg failed ({st})")));
    }
    Ok(())
}
