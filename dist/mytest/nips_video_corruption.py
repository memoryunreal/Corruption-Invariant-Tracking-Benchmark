# Whole Video
def bit_error(src,dst,severity):
    c=[100000, 50000, 30000, 20000, 10000][severity-1]
    return_code = subprocess.run(
        ["ffmpeg","-y", "-i", src, "-c", "copy", "-bsf", "noise={}".format(str(c)),
         dst])

    return return_code



def frame_rate(src,dst,severity):

    c=[10,8,6,4,2][severity-1]
    return_code = subprocess.call(
        ["ffmpeg","-y",  "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-fps", str(c), dst])
    return return_code

# def h264_crf(src,dst,severity):
#     c=[23,30,37,44,51][severity-1]
#     return_code = subprocess.call(
#         ["ffmpeg", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-crf", str(c), dst])

#     return return_code

# def h264_abr(src,dst,severity):

#     c=[2,4,8,16,32][severity-1]
#     result = subprocess.Popen(
#         ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT)

#     data = json.load(result.stdout)

#     bit_rate = str(int(float(data['format']['bit_rate']) / c))

#     return_code = subprocess.call(
#         ["ffmpeg","-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
#          bit_rate, dst])

#     return return_code

# def h265_crf(src, dst, severity):
#     c = [27, 33, 39, 45, 51][severity - 1]
#     return_code = subprocess.call(
#         ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'","-vcodec", "libx265", "-crf", str(c), dst])

#     return return_code

# def h265_abr(src, dst, severity):
#     c = [2, 4, 8, 16, 32][severity - 1]
#     result = subprocess.Popen(
#         ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT)

#     data = json.load(result.stdout)

#     bit_rate = str(int(float(data['format']['bit_rate']) / c))

#     return_code = subprocess.call(
#         ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
#          bit_rate, dst])

#     return return_code