import math
import numpy as np
import statistics as stats

def moving_average(x, window_size=3):
    """ Compute a moving average.

    Example 1: moving_average([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], window_size=3) returns
        [(1 + 2 + 3) / 3, (2 + 3 + 4) / 3, (3 + 4 + 5) / 3, (4 + 5 + 6) / 3]
        which is [2.0, 3.0, 4.0, 5.0].

    Example 2: moving_average([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], window_size=5) returns
        [(1 + 2 + 3 + 4 + 5) / 5, (2 + 3 + 4 + 5 + 6) / 5]
        which is [3.0, 4.0].

    Args:
        x: A list of floats.
        window_size: A positive, odd integer.

    Returns:
        A list of floats.
    """
    if window_size % 2 != 1:
        raise ValueError('window_size must be odd.')
    if window_size > len(x):
        raise ValueError('window_size should be smaller than len(x).')

    # TODO: Replace with valid code.
    y = None
    y = []
    sum = [0]

    for i, xi in enumerate(x, 1):
      sum.append(sum[i-1] + xi)
      if i>=window_size:
        mov_ave = (sum[i] - sum[i-window_size])/window_size
        y.append(mov_ave)

    return y


# In[65]:


def padded_moving_average(x, window_size=3):
    """ Compute a moving average.

    This differs from moving_average in that the input is first
    padded on both sides with an appropriate number of 0s, so that
    the output has the same length as x and so that x and y are
    aligned.

    Example: padded_moving_average([1.0, 1.0, 1.0], window_size=3) returns
        [(0 + 1 + 1) / 3, (1 + 1 + 1) / 3, (1 + 1 + 0) / 3]
        which has approximate values of [0.66, 1.0, 0.66].

    Args:
        x: A list of floats.
        window_size: A positive, odd integer that's less than the length of .

    Returns:
        A list of floats.
    """
    # TODO: Replace with valid code
    y = None
    ba = int((window_size-1)/2)
    x = [0]*ba + x + [0]*ba
    y = moving_average(x, window_size)
    return y


# In[66]:


def filterByStd(x, y1, y2, y3):
    newx=[]
    newy1=[]
    newy2=[]
    newy3=[]
    sig = np.std(x)
    m = np.mean(x)
    for i in range(len(x)):
        if x[i] < m + 3*sig and x[i] > m - 3*sig:
            newx.append(x[i])
            newy1.append(y1[i])
            newy2.append(y2[i])
            newy3.append(y3[i])
    return newx, newy1, newy2, newy3


# In[67]:


def filterByIQR(x,y1,y2,y3):
    '''
    filter by x(range)
    return output in time order(y1)
    '''
    newx=[]
    newy1=[]
    newy2=[]
    newy3=[]
    index = np.argsort(x)
    x.sort()
    q1,q3= np.percentile(x,[25,75])
    print(q1)
    print(q3)
    iqr = q3 - q1
    lower_bound = q1 -(3 * iqr)
    upper_bound = q3 +(3 * iqr)
    for i in range(len(x)):
        if x[i] >= lower_bound and x[i] <= upper_bound:
            newx.append(x[i])
            newy1.append(y1[index[i]])
            newy2.append(y2[index[i]])
            newy3.append(y3[index[i]])
    tindex = np.argsort(newy1)
    nx=[]
    ny2=[]
    ny3=[]
    newy1.sort()
    for i in range(len(newy1)):
        nx.append(newx[tindex[i]])
        ny2.append(newy2[tindex[i]])
        ny3.append(newy3[tindex[i]])
    return nx,newy1,ny2,ny3


def toEuc(r, theta):
    return [r* math.cos(theta),r* math.sin(theta)]



def centroid(pos):
    length = len(pos)
    x_loc = []
    y_loc = []
    if length == 0:
        return None
    for i in range(length):
        x_loc.append(pos[i][0])
        y_loc.append(pos[i][1])
    return [stats.median(x_loc), stats.median(y_loc)]


# In[71]:


def dist(loc1,loc2):
    if loc1 == None or loc2 == None:
        return 0
    return math.sqrt(math.pow(loc2[0]-loc1[0], 2) + math.pow(loc2[1]-loc1[1], 2))

def speedEstimation(pos, time):
    length = len(pos)
    velocity = []
    t = time[0]
    loc = []
    newt = []
    instantRange=[]
    for i in range(length):
        if time[i] == t:
            instantRange.append(pos[i])
        else:
            c = centroid(instantRange)
            newt.append(t)
            loc.append(c)
            instantRange = [pos[i]]
            t = time[i]
   # print(loc)
   # print(newt)
    for i in range(len(loc)):
        if not i == 0:
            velocity.append(dist(loc[i],loc[i-1])/(newt[i]-newt[i-1]))
    return velocity

def averageSpeed(velocity):
    return stats.median(velocity)
