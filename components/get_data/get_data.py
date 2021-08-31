import datetime
import youtube_dl
from subprocess import call
from time import sleep


directory_save = '/home/alina/PycharmProjects/roads/new_data/not-static/'
file_name = '/home/alina/PycharmProjects/roads/components/get_data/data_urls.txt'
ydl = youtube_dl.YoutubeDL()
stream = False


def download_stream_part(url, i, file):
    """Using the youtube-dl library, it downloads an excerpt of a 3-second video stream.
    The video file is saved in the created directory"""

    print("---------download---------")

    result = ydl.extract_info(url, download=False)
    file_path = directory_save + file + '___video' + str(i) + '.mp4'
    command2 = 'ffmpeg -i ' + result['url'] + ' -t 00:00:05.00 -c copy ' + file_path
    call(command2.split(), shell=False)


def get_stream(start_time):
    """For all url from the urls.txt file, the function download is called 10 times"""

    numb = 5
    with open(file_name, 'r') as f:
        for url in f:
            file = start_time + '___' + str(numb)
            numb += 1
            for i in range(10):
                try:
                    download_stream_part(url, i, file)
                    sleep(10)
                except:
                    print('file was not appended')
                    next


def get_video():
    numb = 101
    with open(file_name, 'r') as f:
        for url in f:
            numb += 1
            file_path = directory_save + str(numb) + '___video_only' + '.mp4'
            command2 = 'youtube-dl -o ' + file_path + ' ' + url
            call(command2.split(), shell=False)


if __name__ == '__main__':

    if stream:
        start_time = datetime.datetime.now().strftime('%H')
        get_stream(start_time)
    else:
        get_video()
