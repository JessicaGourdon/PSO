#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:48:09 2020

@author: julien
"""


import numpy as np
import random
from numpy.linalg import *
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as mths
import copy as cp


borneInf = -100
borneSup = 100
Population = 1000
cas = 100
dimension = 2
nbIteration = 5000


def initialise(n,cas,borneInf,borneSup):
    x = np.zeros((n,cas))
    for i in range(n):
        for j in range(cas):
            x[i,j] = np.random.randint(borneInf, borneSup)
    return x

x0 = initialise(dimension,cas,borneInf,borneSup)
v0 = np.zeros((dimension,cas))

def Rosenbrock(E):
    f = []
    n = np.shape(E)[0]
    for i in range(n-1):
        fi = 100*(E[1]-E[0]**2)**2+(1-E[0])**2
        f.append(fi)
    f=np.asarray(f)
    return f

f = Rosenbrock(x0)
#print(x0)
#print(f)
#x=np.min(f)
#print("min",x)
#loc = np.where(f==x)
#loc=loc[1]
#print(loc)
#x1=x0[:,loc]
#print(x1)


def barycentre(M):
    composantes=np.shape(M)[0]
    candidats=np.shape(M)[1]
    bary=np.zeros(composantes)
    x=0
    for j in range(composantes):
        for k in range(candidats):
            x=x+M[j,k]
        x=x/candidats
        bary[j]=x
        
    return (bary)

def compareBarycentre(baryAncien,baryNouveau,epsilon):
    #print(baryAncien)
    #print(baryNouveau)
    norme=mths.sqrt(np.sum((baryNouveau-baryAncien)**2))
    return (norme>epsilon)

TestA=np.ones((2,3))
TestB=15*np.eye(2,3)
baryA=barycentre(TestA)
baryB=barycentre(TestB)
print(compareBarycentre(baryA, baryB, 10E-6))

print(random.random())

mini=10

def PSO(E, f, nbIteration, w, phi1, phi2):
    # Initialisation
    nb = np.shape(E)[1]
    v0 = np.zeros(np.shape(E))
    xAncien=cp.deepcopy(E)
    #print("somme Xancien ", np.sum(xAncien))
    xlb= cp.deepcopy(E)
    flb = f(xlb)
    gb=np.min(flb)
    index = np.where(flb==gb)
    index = index[1]
    xGB = xlb[:,index]
    k = 1
    vAncien=v0
    vSuiv = vAncien+phi2*random.random()*(xGB-x0)
    xSuiv = xAncien+vSuiv
    #print("somme Xsuivant ", np.sum(xSuiv))
    #print("somme Xancien apres création suivant ", np.sum(xAncien))
    for i in range(nb):
        
        fxi=f(xSuiv[:,i])

        #print("somme Xancien apres range fxi et i : ",i, np.sum(xAncien))
        if ((fxi < flb).any()):
            xlb[:,i] = xSuiv[:,i]
    #print("somme Xancien apres range fxi ", np.sum(xAncien))
    flb = f(xlb)
    gbNouveau=np.min(flb)
    #print("iiii")
    if (gbNouveau < gb):
        gb=gbNouveau
        index = np.where(flb==gb)
        index = index[1]
        xGB = xlb[:,index]
    
    
    #print(compareBarycentre(barycentre(xAncien), barycentre(xSuiv), 10E-6))
    #print(np.sum(xAncien))
    #print(np.sum(xSuiv))
    #print("zzz")
    # Boucle
    while  (k<=nbIteration and compareBarycentre(barycentre(xAncien), barycentre(xSuiv), 10E-10)) :
        #print("eee")
        vAncien=cp.deepcopy(vSuiv)
        xAncien=cp.deepcopy(xSuiv)
        vSuiv = vAncien+phi1*random.random()*(xlb-xAncien)+phi2*random.random()*(xGB-xAncien)
        xSuiv = xAncien+vSuiv
        for i in range(nb):
            fxi=f(xSuiv[:,i])
            if ((fxi < flb).any()):
                xlb[:,i] = xSuiv[:,i]
        flb = f(xlb)
        gbNouveau=np.min(flb)
        if (gbNouveau < gb):
            gb=gbNouveau
            index = np.where(flb==gb)
            index = index[1]
            xGB = xlb[:,index]

        k=k+1
        

    print(k)
    xMini = xGB
    fMini=f(xMini)
    return xMini,fMini

xMini,fMini=PSO(x0, Rosenbrock, nbIteration, 1, 1, 1.5)
print("valeur de x min :",xMini)
print("Valeur de f min : ",fMini)
