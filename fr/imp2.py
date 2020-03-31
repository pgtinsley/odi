#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import face_recognition
from scipy import stats
import matplotlib.pyplot as plt


# In[2]:


with open('./__real_encodings.pkl', 'rb') as f:
    real_encodings_dict = pickle.load(f)
with open('./__generated_encodings.pkl', 'rb') as f2:
    generated_encodings_dict = pickle.load(f2)


# In[3]:


real_encodings_list = []
for i, d in real_encodings_dict.items():
    real_encodings_list.append(d['enc'])


# In[4]:


generated_encodings_list = []
for i, d in generated_encodings_dict.items():
    generated_encodings_list.append(d['enc'])


# In[14]:


# dist_mat_imp1 = []
# for enc in real_encodings_list:    
#     # dist_mat.append(face_recognition.face_distance(real, enc))
#     dist_mat_imp1.append(face_recognition.face_distance(generated_encodings_list, enc))


# In[32]:


# dist_list_imp1 =  []
# for i, arr in enumerate(dist_mat_imp1):
#     for e in arr[:i]:
# #     for e in arr[:i+1]:
#         dist_list_imp1.append(e)
# 
# with open('dist_list_imp1.pkl', 'wb') as f:
#     pickle.dump(dist_list_imp1, f)

# In[29]:


dist_mat_imp2 = []
<<<<<<< HEAD
for i, enc in enumerate(real_encodings_list):
    print(i)   
    dist_mat_imp2.append(face_recognition.face_distance(real_encodings_list, enc))
    # dist_mat_imp1.append(face_recognition.face_distance(generated_encodings_list, enc))z

# In[30]:


# dist_list_imp2 =  []
# for i, arr in enumerate(dist_mat_imp2):
#     for e in arr[:i]:
#         dist_list_imp2.append(e)
# 
# with open('dist_list_imp2.pkl', 'wb') as f:
#     pickle.dump(dist_list_imp2, f)
=======
for enc in real_encodings_list:    
    dist_mat_imp2.append(face_recognition.face_distance(real_encodings_list, enc))
    # dist_mat_imp1.append(face_recognition.face_distance(generated_encodings_list, enc))


# In[30]:
>>>>>>> 3a73d11a261645630e6422a9597702bf97b5134d


dist_list_imp2 =  []
for i, arr in enumerate(dist_mat_imp2):
    for e in arr[:i]:
        dist_list_imp2.append(e)

<<<<<<< HEAD
    if (i % 7000 == 0) | (i == len(dist_mat_imp2)-1):    
        with open('dist_list_imp2_{}.pkl', 'wb') as f:
            pickle.dump(dist_list_imp2, f)
        dist_list_imp2 = []
=======
with open('dist_list_imp2.pkl', 'wb') as f:
    pickle.dump(dist_list_imp2, f)
>>>>>>> 3a73d11a261645630e6422a9597702bf97b5134d


# In[37]:


# plt.figure(figsize=(10,7), facecolor='white')
# plt.title('Impostor Distributions')
# plt.hist(dist_list_imp1, bins='auto', color='red', alpha=0.5, label='real vs. generated')
# plt.hist(dist_list_imp2, bins='auto', color='blue', alpha=0.5, label='real vs. real')
# plt.xlabel('Distance between Face Encodings')
# plt.ylabel('Count')
# plt.legend()
# plt.show()


# In[46]:


# s, p = stats.ks_2samp(dist_list_imp1, dist_list_imp2)
# s, p


# In[45]:


# null hypothesis = 2 independent samples are drawn from the same continuous distribution


# In[47]:


# p < 0.05, so we reject the null hypothesis --> data is drawn from different distributions


# In[ ]:




