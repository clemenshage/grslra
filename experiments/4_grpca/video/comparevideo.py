import subprocess

# This helper script is based on ffmpeg and creates tiled video to compare the results of four background subtraction algorithms

outfilename = 'comparison_p'

algolist = ['k4_p0_1', 'k4_p0_4', 'k4_p0_7', 'k4_p1_0']
#algolist = ['k4_p0_1', 'ialm', 'lmafit', 'godec']

bg_filelist = []
fg_filelist = []

for algo in algolist:
    bg_filelist.append('background_' + algo + '.mpg')
    fg_filelist.append('foreground_' + algo + '.mpg')


cols = 160
rows = 130
videoscaling = 2

scalestr = '{:d}x{:d}'.format(videoscaling * cols, videoscaling * rows)
fullscalestr = '{:d}x{:d}'.format(2 * videoscaling * cols, 2 * videoscaling * rows)

fg_outfilename = 'fg_' + outfilename + '.mkv'
subprocess.call(str().join(['ffmpeg -i ', str(' -i ').join(fg_filelist), ' -filter_complex "nullsrc=size=', fullscalestr, '[base]; [0:v] setpts=PTS-STARTPTS, scale=', scalestr, '[upperleft]; [1:v] setpts=PTS-STARTPTS, scale=', scalestr, '[upperright]; [2:v] setpts=PTS-STARTPTS, scale=', scalestr, '[lowerleft]; [3:v] setpts=PTS-STARTPTS, scale=', scalestr, '[lowerright]; [base][upperleft] overlay=shortest=1 [tmp1]; [tmp1][upperright] overlay=shortest=1:x=', '{:d}'.format(videoscaling*cols), '[tmp2]; [tmp2][lowerleft] overlay=shortest=1:y=', '{:d}'.format(videoscaling*rows), '[tmp3]; [tmp3][lowerright] overlay=shortest=1:x=', '{:d}'.format(videoscaling*cols), ':y=', '{:d}'.format(videoscaling*rows), '" -c:v libx264 -y ', fg_outfilename]),shell=True)

bg_outfilename = 'bg_' + outfilename + '.mkv'
subprocess.call(str().join(['ffmpeg -i ', str(' -i ').join(bg_filelist), ' -filter_complex "nullsrc=size=', fullscalestr, '[base]; [0:v] setpts=PTS-STARTPTS, scale=', scalestr, '[upperleft]; [1:v] setpts=PTS-STARTPTS, scale=', scalestr, '[upperright]; [2:v] setpts=PTS-STARTPTS, scale=', scalestr, '[lowerleft]; [3:v] setpts=PTS-STARTPTS, scale=', scalestr, '[lowerright]; [base][upperleft] overlay=shortest=1 [tmp1]; [tmp1][upperright] overlay=shortest=1:x=', '{:d}'.format(videoscaling*cols), '[tmp2]; [tmp2][lowerleft] overlay=shortest=1:y=', '{:d}'.format(videoscaling*rows), '[tmp3]; [tmp3][lowerright] overlay=shortest=1:x=', '{:d}'.format(videoscaling*cols), ':y=', '{:d}'.format(videoscaling*rows), '" -c:v libx264 -y ', bg_outfilename]),shell=True)