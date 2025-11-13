for dir in data/video2img_2fps/*_*; do
    folder_name=$(basename "$dir")
    output_video="${dir}/output.mp4"

    echo "ffmpeg -framerate 2 -i "${dir}/%04d.png" -c:v libx264 -pix_fmt yuv420p "${output_video}""
    # ffmpeg -framerate 2 -pattern_type glob -i "${dir}/*.png" -c:v libx264 -pix_fmt yuv420p "${output_video}"
    echo "${output_video},0" >> video_list.csv
done