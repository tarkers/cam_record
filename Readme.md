### ffmpeg lower resolution
ffmpeg -i input.avi -vf scale="720:480" output.avi
### ffmpeg change extension

ffmpeg -i my_mkv.mkv -codec copy my_mkv.mp4