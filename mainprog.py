import sys
import subprocess
import pkg_resources
subprocess.run([sys.executable,"-m", 'apt' ,'install' ,'ffmpeg','streamlit','librosa','numba'])
proc = subprocess.Popen('pip install numba',
                        shell=True, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
proc = subprocess.Popen('pip install spleeter',
                        shell=True, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
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
# Helper functions
def tree(dir_path: Path, prefix: str = ''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix + extension)
def get_dirs_inside_dir(folder):
    return [my_dir for my_dir in list(map(lambda x:os.path.basename(x), sorted(Path(folder).iterdir(), key=os.path.getmtime, reverse=True))) if os.path.isdir(os.path.join(folder, my_dir))
            and my_dir != '__pycache__' and my_dir != '.ipynb_checkpoints' and my_dir != 'API']
def list_folders_in_folder(folder):
    return [file for file in os.listdir(folder) if os.path.isdir(os.path.join(folder, file))]
def show_dir_tree(folder):
    with st.expander(f"Show {os.path.basename(folder)} folder tree"):
        for line in tree(Path.home() / folder):
            st.write(line)
def delete_folder(folder, ask=True):
    if not ask:
        shutil.rmtree(folder)
    else:
        folder_basename = os.path.basename(folder)
        if len(os.listdir(folder)) > 0:
            st.warning(f"**{folder_basename} is not empty. Are you sure you want to delete it?**")
            show_dir_tree(folder)
            if st.button("Yes"):
                try:
                    shutil.rmtree(folder)
                except:
                    st.error(f"Couldn't delete {folder_basename}:")
                    e = sys.exc_info()
                    st.error(e)
        else:
            st.write(f"**Are you sure you want to delete {folder_basename}?**")
            if st.button("Yes"):
                try:
                    shutil.rmtree(folder)
                except:
                    st.error(f"Couldn't delete {folder_basename}:")
                    e = sys.exc_info()
                    st.error(e)
                    
# Implementation
    col1_size = 10
    col1, col2 = st.columns((col1_size, 1))
    with col1:
        models_abs_dir = os.path.join(configs['APP_BASE_DIR'], configs['MODELS_DIR'])
        temp = []
        i = 0
        while temp != configs['CURRNET_FOLDER_STR'] and temp != configs['CREATE_FOLDER_STR']:
            i += 1
            state.files_to_show = get_dirs_inside_dir(models_abs_dir)
            temp = st.selectbox("Models' folder" + f": level {i}",
                                options=[configs['CURRNET_FOLDER_STR']] + state.files_to_show
                                        + [configs['CREATE_FOLDER_STR']] + [configs['DELETE_FOLDER_STR']],
                                key=models_abs_dir)
            if temp == configs['CREATE_FOLDER_STR']:
                new_folder = st.text_input(label="New folder name", value=str(state.dataset_name) + '_' +
                                                                          str(state.model) + '_models', key="new_folder")
                new_folder = os.path.join(models_abs_dir, new_folder)
                if st.button("Create new folder"):
                    os.mkdir(new_folder)
                    state.files_to_show = get_dirs_inside_dir(models_abs_dir)
            elif temp == configs['DELETE_FOLDER_STR']:
                if list_folders_in_folder(models_abs_dir):
                    chosen_delete_folder = st.selectbox(
                        label="Folder to delete",                        options=list_folders_in_folder(models_abs_dir), key="delete_folders")
                    chosen_delete_folder = os.path.join(models_abs_dir, chosen_delete_folder)
                    delete_folder(chosen_delete_folder)
                    state.files_to_show = get_dirs_inside_dir(models_abs_dir)
                else:
                    st.info('No folders found')
            elif not temp == configs['CURRNET_FOLDER_STR']:
                models_abs_dir = os.path.join(models_abs_dir, temp)
        try:
            show_dir_tree(models_abs_dir)
        except FileNotFoundError:
            pass
        table = st.empty()
        try:
            files_in_dir = os.listdir(models_abs_dir)
            if ".gitignore" in files_in_dir:
                files_in_dir.remove(".gitignore")
            table.write(models_table(files_in_dir))
        except FileNotFoundError:
            st.error("No 'saved_models' folder, you should change working dir.")
        except ValueError:
            pass
        except:
            e = sys.exc_info()
            st.info(e)
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
    new_file = Path(f'{base}.wav')
    os.rename(out_file, new_file)
    ##@ Check success of download
    if new_file.exists():
        print(f'{yt.title} has been successfully downloaded.')
        idsave=fname
        fnamesave=fname+'.wav'
          #--------------------------------------------------
        fext=cwd+"/audio/"+fname+'/'
          #--------------------------------------------------
        fname=cwd+"/audio/"+fname+'/'+fname+'.wav'
        out=cwd+'/audio/'
        subprocess.run(["spleeter", "separate", fname ,"-p" "spleeter:5stems", "-c", "wav", "-o", out], capture_output=True)
        #--------------------------------------------------
        dfinfo=ytdata(url)
        df1=extract_features_orig(fname)
        
        user_input2=st.text_input(cwd)
        df2=extract_features_spleeted(fext+'vocals.wav','vocals')
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
