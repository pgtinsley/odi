#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import pickle
import face_recognition


# In[ ]:


real_70000 = glob.glob('../real_70000_ffhq/*.png')
generated_70000 = glob.glob('../generated_70000_sg2/*.png')


# In[ ]:


# len(real_70000), len(generated_1200)


# In[ ]:


# get 1000 encodings from generated_1200
generated_encodings = {}
counter = 0
for i, fname in enumerate(generated_70000):
    print('Loading image {}/{}...'.format(i, 69999))
    encs = face_recognition.face_encodings(
        face_recognition.load_image_file(fname)
    )
    if len(encs) > 0: 
        generated_encodings[counter] = {
            'fname_generated_70000': fname.split('/')[-1],
            'enc': encs[0]
        }
        counter += 1
    else: print(f'No face found in image {i}...')


# In[ ]:


with open('generated_encodings.pkl', 'wb') as f:
    pickle.dump(generated_encodings, f)

