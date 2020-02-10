#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import os

from dataset_utils import create_rgb_array_lozenge, show_center


# In[4]:


# matplotlib parameters for red balls plot
#plt.rcParams['figure.dpi'] = 200
#plt.rcParams['axes.grid'] = False
#plt.rcParams['axes.axisbelow'] = False


# ## Dataset

# In[5]:


# parameters
size = (64, 64)


# ### Training

# In[88]:


# generate 30000 training images
n_samples = 30000

images_train = np.zeros((n_samples, *size, 3))
centers_train = np.column_stack([np.random.randint(0, 63, n_samples), np.random.randint(0, 63, n_samples)])
scales_train = 10 * np.random.rand(n_samples) + 7
rots_train = np.pi * np.random.rand(n_samples)
for i, (center, scale, rot) in tqdm(enumerate(zip(centers_train, scales_train, rots_train))):
    images_train[i] = create_rgb_array_lozenge(center=center, size=size, scale=scale, rot=rot)


# In[89]:


images_train.shape


# In[1]:


# test print center as white pixel
#idx = np.random.choice(len(images_train), 20)
#show_center(images_train[idx], centers_train[idx])


# ### Testing

# In[91]:


# generate 10000 testing images
n_samples = 10000
images_test = np.zeros((n_samples, *size, 3))
centers_test = np.column_stack([np.random.randint(0, 63, n_samples), np.random.randint(0, 63, n_samples)])
scales_test = 20 * np.random.rand(n_samples) + 7
rots_test = np.pi * np.random.rand(n_samples)
for i, (center, scale, rot) in tqdm(enumerate(zip(centers_test, scales_test, rots_test))):
    images_test[i] = create_rgb_array_lozenge(center=center, size=size, scale=scale, rot=rot)


# In[92]:


# test print center as white pixel
#idx = np.random.choice(len(images_test), 20)
#show_center(images_test[idx], centers_test[idx])


# In[93]:

if os.path.exists('data_loader/data/red_lozenges/'):
    print('Data folder exists')
else:
    print('Data folder does not exist, making it')
    os.mkdir('data_loader/data/red_lozenges')
np.save('data_loader/data/red_lozenges/images_train', images_train)
np.save('data_loader/data/red_lozenges/centers_train', centers_train)
np.save('data_loader/data/red_lozenges/scales_train', scales_train)
np.save('data_loader/data/red_lozenges/rots_train', rots_train)
np.save('data_loader/data/red_lozenges/images_test', images_test)
np.save('data_loader/data/red_lozenges/centers_test', centers_test)
np.save('data_loader/data/red_lozenges/scales_test', scales_test)
np.save('data_loader/data/red_lozenges/rots_test', rots_test)
print('done')
