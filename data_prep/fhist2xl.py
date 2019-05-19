import scipy.io as sio
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
from collections import Counter
import math
import statistics as stats
import functions
import os
import csv

for filename in os.listdir('fhist_data'):
    f1 = './fhist_data/'+filename
    f = sio.loadmat(f1)
    fhist = f['fHist']
    pc = fhist['pointCloud']
    header= fhist['header']
    ts = fhist['timestamp']
    num_points = fhist['numInputPoints']
    dp=[]
    snr=[]
    r=[]
    time=[]
    angle=[]
    pos=[]
    rangeSTD=[]
    normalizedRangeSTD=[]
    midRangeIntensity=[]
    NumberPointcloud=[]
    frameTime=[]
    SumSNR=[]
    SpeedPerFrame=[]
    MeanAngle=[]
    angleSTD=[]


    #fhist.shape[1]
    for i in range(fhist.shape[1]):
        #calculate the standard deviations and mean of points range data
        rdata = []
        for k in range(num_points[0,i][0,0]):
            rdata.append(pc[0,i][0,k])
        if len(rdata)!=0:
            sig = np.std(rdata)
            m = np.mean(rdata)
        
        #find range, snr, time, doppler value of data points
        temp=[]
        pointsCollected = False
        for j in range(num_points[0,i][0,0]):
            #clean the data
            if pc[0,i][0,j] > m + 3*sig or pc[0,i][0,j] < m - 3*sig:
                continue

            if pc[0,i][3,j] <= 0:
                snr.append(0.00000000001)
            if pc[0,i][3,j] > 0:
                snr.append(10*np.log10(pc[0,i][3,j]))
    #     t = header[0,i]['timestamp'][0,0][0][0]
            dp.append(pc[0,i][2,j])
            r.append(pc[0,i][0,j])
            angle.append(pc[0,i][1,j])
            time.append(i/20)
    #     s = toEuc(pc[0,i][0,j],pc[0,i][1,j])
            pos.append(functions.toEuc(pc[0,i][0,j],pc[0,i][1,j]))
            temp.append(pc[0,i][0,j])
            medianIndex = np.argsort(temp)[len(temp)//2]
            pointsCollected = True
        if pointsCollected:
            frameTime.append(i/20)
            rangeSTD.append(np.std(temp))
            angleSTD.append(np.std(pc[0,i][1,:]))
            normalizedRangeSTD.append(np.std(temp)*np.sqrt(stats.median(temp)))
            midRangeIntensity.append(10*np.log10(pc[0,i][3,medianIndex]))
            NumberPointcloud.append(num_points[0,i][0,0])
            SumSNR.append(sum(10*np.log10(pc[0,i][3,:])))
            SpeedPerFrame = functions.speedEstimation(pos, time)
            MeanAngle.append(np.mean(pc[0,i][1,:]))

    #snr = functions.padded_moving_average(snr,3)
    normalizedRangeSTD = functions.padded_moving_average(normalizedRangeSTD, 7)
    fn = filename[0:-3]+'csv'
    with open(fn, mode='w') as myfile:
        w = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        w.writerow(time)
        w.writerow(dp)
        w.writerow(angle)
        w.writerow(snr)
        w.writerow(frameTime)
        w.writerow(rangeSTD)
        w.writerow(normalizedRangeSTD)
        w.writerow(NumberPointcloud)
        w.writerow(MeanAngle)
