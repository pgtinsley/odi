#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pickle
import face_recognition


# In[ ]:


real_70000 = glob.glob('../real_70000_ffhq/*.png')
generated_1200 = glob.glob('../generated_1200_sg2/*.png')


# In[ ]:


len(real_70000), len(generated_1200)


# In[ ]:


# get 1000 encodings from real_70000
real_encodings = {}
counter = 0
for i, fname in enumerate(real_70000):
    print('Loading image {}/{}...'.format(i, 69999))
    encs = face_recognition.face_encodings(
        face_recognition.load_image_file(fname)
    )
    if len(encs) > 0: 
        real_encodings[counter] = {
            'fname_real_70000': fname.split('/')[-1],
            'enc': encs[0]
        }
        counter += 1
    else: print(f'No face found in image {i}...')

# In[ ]:


with open('real_encodings.pkl', 'wb') as f:
    pickle.dump(real_encodings, f)

