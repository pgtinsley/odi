#!/usr/bin/env python
# coding: utf-8

# In[3]:


import glob
import pickle
from PIL import Image
import face_recognition


# In[4]:


# fnames = glob.glob('../real_70000_ffhq/*.png')
fnames = glob.glob('../../stylegan2/results/00015-generate-images/*.png')


# In[7]:


for fname in fnames:
    img = face_recognition.load_image_file(fname)
    locations = face_recognition.face_locations(img)
    if len(locations)>0:
        (top, right, bottom, left) = locations[0]
        chip = Image.fromarray(img[top: bottom, left: right])
        chip.save('../generated_70000_sg2_chips/{}'.format(fname.split('/')[-1]))
    else: print('No face found in image {}'.format(fname))


# In[4]:


# with open('./generated_encodings.pkl', 'rb') as f:
#     generated_encodings_dict = pickle.load(f)


# In[7]:


# for i, d in generated_encodings_dict.items():
#     img = face_recognition.load_image_file('../generated_1200_sg2/' + d['fname_generated_1000'])
#     (top, right, bottom, left) = face_recognition.face_locations(img)[0]
#     chip = Image.fromarray(img[top: bottom, left: right])
#     chip = chip.save('../generated_1000_sg2_chips/{}'.format(d['fname_generated_1000']))


# In[ ]:




