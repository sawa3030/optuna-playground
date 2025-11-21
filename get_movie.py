from moviepy import ImageSequenceClip
import glob

# 保存した画像を読み込む
image_files = sorted(glob.glob("frames/frame_*.png"))

# fps = コマ送りの速さ（1秒あたり何枚）
clip = ImageSequenceClip(image_files, fps=2)

# mp4 で保存
clip.write_videofile("contour_animation.mp4", codec="libx264")
