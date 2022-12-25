import sys
import subprocess
import pkg_resources
subprocess.run([sys.executable,"-m", 'apt' ,'install' ,'ffmpeg','streamlit','librosa','numba'])
proc = subprocess.Popen('pip install numba',
                        shell=True, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spleeter'])
proc = subprocess.Popen('pip install pathlib',
                        shell=True, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
proc = subprocess.Popen('pip install pytube3 --upgrade',
                        shell=True, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
proc = subprocess.Popen('pip install urllib',
                        shell=True, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
proc = subprocess.Popen('-m pip install spleeter',
                        shell=True, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)





from pytube import YouTube
from pathlib import Path
import os
from os.path import basename
import time
import spleeter
import librosa
import numpy as np
import pandas as pd
import urllib.request
from PIL import Image
import io
import streamlit as st
import ffmpeg

cwd=str(os.getcwd())
subprocess.run(["git", "clone", "https://github.com/iwantthatresult/ytdlspleeter.git"])
gitdir=cwd+'ytdlspleeter'
def save(fname,TOKEN):
  savecwd=cwd
  os.chdir(cwd +'/ytdlspleeter')
  subprocess.run(['git','remote', 'set-url', 'origin', 'https://iwantthatresult:'+TOKEN+'+"@github.com/iwantthatresult/ytdlspleeter.git'])
  subprocess.run(['mv' ,savecwd+'/audio/'+fname, './data'])
  subprocess.run(['git','add', './data/'+fname])
  subprocess.run(['git','config','user.email', '"space.punk3r@gmail.com"'])
  subprocess.run(['git','config','user.name', '"iwantthatresult"'])
  subprocess.run(['git', 'commit', '-m', '"adding new song"'])
  subprocess.run(['git' ,'push',  'https://iwantthatresult:'+TOKEN+'@github.com/iwantthatresult/ytdlspleeter.git'])
  os.chdir(savecwd)
#--------------------------------------------------
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            st.write(subindent +' // ' + f + ' // ' + os.path.basename(root))
            
def list_directories(startpath):
    for root, dirs, files in os.walk(startpath):
        st.write(root)
        list_files(root)




def get_file_list(directory):
    # Create an empty list to store the names of the files
    file_list = []

    # Use the os.listdir() function to get a list of all the files in the directory
    for filename in os.listdir(directory):
        file_list.append(filename)

    return file_list

def compare_lists(list1, list2):
    # Create an empty list to store the names of any new files
    new_files = []

    # Iterate over both lists and compare the elements
    for file in list1:
        if file not in list2:
            new_files.append(file)

    # If there are any new files, print their names
    if len(new_files) > 0:
        print("New files:")
        for file in new_files:
           st.write(file)
    else:
        st.write("No new files")



def extract_features_orig(file_path,total=False):
  # Load the audio file
  audio, sample_rate = librosa.load(file_path)
  audio, _ = librosa.effects.trim(audio, top_db=60)
    # Extract 17 features using librosa
  features=[]
  features.append([audio])
  features.append(librosa.beat.tempo(audio, sample_rate)[0])
  features.append(librosa.feature.chroma_stft(audio, sample_rate))
  features.append(librosa.feature.chroma_cqt(audio, sample_rate))
  features.append(librosa.feature.chroma_cens(audio, sample_rate))
  features.append(librosa.feature.mfcc(audio, sample_rate))
  features.append(librosa.feature.spectral_centroid(audio, sample_rate))
  features.append(librosa.feature.spectral_bandwidth(audio, sample_rate))
  features.append(librosa.feature.spectral_rolloff(audio, sample_rate))
  features.append(librosa.feature.spectral_contrast(audio, sample_rate))
  features.append(librosa.feature.tonnetz(audio, sample_rate))
  features.append(librosa.feature.zero_crossing_rate(audio))
  features.append(librosa.feature.poly_features(audio, sample_rate))
  features.append(librosa.feature.tempogram(audio, sample_rate))
  features.append(librosa.feature.spectral_flatness(audio))
  features.append(librosa.beat.beat_track(audio, sample_rate)[1])
  df = pd.DataFrame(data=[features], columns=['audio','BPM','Chromastft', 'Chromacqt', 'Chromacens', 'mfccs', 'spectralcentroid', 'spectral_bandwith', 'spectral_rolloff', 'spectral_contrast', 'tonnetz',
                                     'zero_crossing_rate', 'poly_features', 'tempogram', 'spectral flattness','beatframe'])
  # Return the audio and extracted features
  return df
def extract_features_spleeted(file_path,spleet):
  # Load the audio file
  audio, sample_rate = librosa.load(file_path)
    # Extract 17 features using librosa
  features=[]
  features.append([audio])
  features.append(librosa.feature.chroma_stft(audio, sample_rate))
  features.append(librosa.feature.chroma_cqt(audio, sample_rate))
  features.append(librosa.feature.chroma_cens(audio, sample_rate))
  features.append(librosa.feature.mfcc(audio, sample_rate))
  features.append(librosa.feature.spectral_centroid(audio, sample_rate))
  features.append(librosa.feature.spectral_bandwidth(audio, sample_rate))
  features.append(librosa.feature.spectral_rolloff(audio, sample_rate))
  features.append(librosa.feature.spectral_contrast(audio, sample_rate))
  features.append(librosa.feature.tonnetz(audio, sample_rate))
  features.append(librosa.feature.zero_crossing_rate(audio))
  features.append(librosa.feature.poly_features(audio, sample_rate))
  features.append(librosa.feature.spectral_flatness(audio))
  if spleet=='drums':
    features.append(librosa.feature.tempogram(audio, sample_rate))
    features.append(librosa.beat.beat_track(audio, sample_rate)[0])
    features.append(librosa.beat.beat_track(audio, sample_rate)[1])
    df = pd.DataFrame(data=[features], columns=['audio '+spleet,'Chromastft ' +spleet, 'Chromacqt ' +spleet, 'Chromacens ' +spleet, 'mfccs ' +spleet, 'spectralcentroid ' +spleet, 'spectral_bandwith ' +spleet, 'spectral_rolloff ' +spleet, 'spectral_contrast ' +spleet, 'tonnetz ' +spleet,
                                     'zero_crossing_rate' +spleet, 'poly_features' +spleet, 'tempogram' +spleet, 'spectral flattness' +spleet,'BPM ' +spleet ,'beatframe' +spleet])
  else:
    df = pd.DataFrame(data=[features], columns=['audio '+spleet,'Chromastft ' +spleet, 'Chromacqt ' +spleet, 'Chromacens ' +spleet, 'mfccs ' +spleet, 'spectralcentroid ' +spleet, 'spectral_bandwith ' +spleet, 'spectral_rolloff ' +spleet, 'spectral_contrast ' +spleet, 'tonnetz ' +spleet,
                                     'zero_crossing_rate' +spleet, 'poly_features' +spleet, 'spectral flattness' +spleet])
  # Return the extracted features and BPM
  return df
def remplacer_caracteres(chaine, ancien_caractere, nouveau_caractere):
  # Utiliser la méthode replace() de la classe str pour remplacer les occurences
  # de l'ancien caractère par le nouveau
  nouvelle_chaine = chaine.replace(ancien_caractere, nouveau_caractere)
  return nouvelle_chaine
def arraysavefromurl(url):
  # Open the URL and read the image data
  with urllib.request.urlopen(url) as url:
    image_data = url.read()
  # Create a file-like object from the image data
  image_file = io.BytesIO(image_data)
  # Open the image using the PIL library
  image = Image.open(image_file)
  # Convert the image to a NumPy array
  image_array = np.array(image)
  return image_array
def ytdata(url):
  video = YouTube(url)
  title=video.title
  title=remplacer_caracteres(title, '\n', ' ')  
  description=video.description
  description=remplacer_caracteres(description, '\n', ' ')
  keywords=video.keywords
  duration=video.length
  views=video.views
  meta=video.metadata
  id=video.video_id
  img=arraysavefromurl(video.thumbnail_url)
  df=pd.DataFrame(data=[[id,title,description,keywords,duration,views,meta]],columns=['id','Title','Description','keywords','duration','views','meta'])
  return df
  #--------------------------------------------------
def youtube2mp3 (url,outdir,fname,Token):
    # url input from user
    yt = YouTube(url)
    ##@ Extract audio with 160kbps quality from video
    video = yt.streams.filter(only_audio=True,abr='160kbps').last()
    ##@ Downloadthe file
    out_file = video.download(output_path=outdir,filename=fname)
    base, ext = os.path.splitext(out_file)
    new_file = Path(f'{base}.mp3')
    os.rename(out_file, new_file)
    ##@ Check success of download
    if new_file.exists():
        print(f'{yt.title} has been successfully downloaded.')
        idsave=fname
        fnamesave=fname+'.mp3'
          #--------------------------------------------------
        fext=cwd+"/audio/"+fname+'/'
          #--------------------------------------------------
        fname=cwd+"/audio/"+fname+'/'+fname+'.mp3'
        out=cwd+'/audio/'
        list1 = get_file_list(out)
        subprocess.run(["spleeter", "separate", fname ,"-p" "spleeter:5stems", "-c", "mp3", "-o", out], capture_output=True)
        list2 = get_file_list(out)
        compare_lists(list1, list2)
        audio_file = open(fname, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='mp3')
        user_input=st.text_input(fname)
        
        list_directories(outdir)
        list_files(outdir)
    
        #--------------------------------------------------
        dfinfo=ytdata(url)
        df1=extract_features_orig(fname)
        st.dataframe(data=df1)        
        user_input2=st.text_input(cwd)
        df2=extract_features_spleeted(out+'vocals.mp3','vocals')
        df3=extract_features_spleeted(fext+'drums.wav','drums')
        df4=extract_features_spleeted(fext+'piano.wav','other')
        df5=extract_features_spleeted(fext+'piano.wav','piano')
        df6=extract_features_spleeted(fext+'bass.wav','bass')
        df=pd.concat([dfinfo,df1,df2,df3,df4,df5,df6],axis=0)
        df.to_csv(fext+idsave+".csv", index=False)
        #----------------------------------------------------------
        save(idsave,Token)
    else:
        print(f'ERROR: {yt.title}could not be downloaded!')
    
def audiodl(id):
  id=str.split(id)
  print(id)
  Token=id[0]
  for i in range(1,len(id)):
    url='www.youtube.com/watch?v='+id[i]
    youtube2mp3(url,cwd+'/audio/'+str(id[i])+"",str(id[i]),Token)  
a='ghp_1DdIbeU8qf02IzgQ0s'+'5PguV5GQRz2w4FP4DT '
b='aO_nmfMc2y4'
a=audiodl(a+b)
user_input=st.text_input(cwd)
