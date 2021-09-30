# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:57:14 2020

@author: sun.fa
"""

import numpy as np
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================
# class of Cubic B-spline basis 
# ======================================
class splineBasis():
    def __init__(self, knots, t_measurement, t_collocation):
        """
        Get the basis matrices and derivative basis matrices of cubic b-spline 
        with given control points
        
        knots: position of control points on time domain
        t_measurement: measurement points on time domain
        t_collocation: collocation points on time domain
        """
        super(splineBasis, self).__init__()
        self.knots = knots
        self.t_m = t_measurement
        self.t_c = t_collocation

    def get_measurement(self):
        """
        construct cubic b spline basis matrix for measurement basis
        """        

        b_spline_0 = np.zeros([len(self.t_m), len(self.knots)-1])
        
        for i in range(b_spline_0.shape[0]):
            for j in range(b_spline_0.shape[1]):
                if self.t_m[i] >= self.knots[j] and self.t_m[i] < self.knots[j+1]:
                    b_spline_0[i, j] = 1
        
        b_spline_1 = np.zeros([len(self.t_m), len(self.knots)-2])
        
        for i in range(b_spline_1.shape[0]):
            for j in range(b_spline_1.shape[1]):
                basis_1 = (self.t_m[i] - self.knots[j]) / (self.knots[j+1] - self.knots[j]) * b_spline_0[i,j]
                basis_2 = (self.knots[j+2] - self.t_m[i]) / (self.knots[j+2] - self.knots[j+1]) * b_spline_0[i,j+1]
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_1[i,j] = basis_1 + basis_2
        
        b_spline_2 = np.zeros([len(self.t_m), len(self.knots)-3])
        
        for i in range(b_spline_2.shape[0]):
            for j in range(b_spline_2.shape[1]):
                basis_1 = (self.t_m[i] - self.knots[j]) / (self.knots[j+2] - self.knots[j]) * b_spline_1[i,j]
                basis_2 = (self.knots[j+3] - self.t_m[i]) / (self.knots[j+3] - self.knots[j+1]) * b_spline_1[i,j+1]
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_2[i,j] = basis_1 + basis_2
                
        b_spline_3 = np.zeros([len(self.t_m), len(self.knots)-4])
        
        for i in range(b_spline_3.shape[0]):
            for j in range(b_spline_3.shape[1]):
                basis_1 = (self.t_m[i] - self.knots[j]) / (self.knots[j+3] - self.knots[j]) * b_spline_2[i,j]
                basis_2 = (self.knots[j+4] - self.t_m[i]) / (self.knots[j+4] - self.knots[j+1]) * b_spline_2[i,j+1]
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_3[i,j] = basis_1 + basis_2
                
        b_spline_3_dt = np.zeros([len(self.t_m), len(self.knots)-4])
        
        for i in range(b_spline_3_dt.shape[0]):
            for j in range(b_spline_3_dt.shape[1]):
                basis_1 = b_spline_2[i,j] / (self.knots[j+3] - self.knots[j])
                basis_2 = b_spline_2[i,j+1] / (self.knots[j+4] - self.knots[j+1])
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_3_dt[i,j] = 3 * (basis_1 - basis_2)
                
        return b_spline_3, b_spline_3_dt
    

    def get_collocation(self):
        """
        construct cubic b spline basis and its derivative basis matrices 
        for collocation domain
        """        
        
        b_spline_0 = np.zeros([len(self.t_c), len(self.knots)-1])
        
        for i in range(b_spline_0.shape[0]):
            for j in range(b_spline_0.shape[1]):
                if self.t_c[i] >= self.knots[j] and self.t_c[i] < self.knots[j+1]:
                    b_spline_0[i, j] = 1
        
        b_spline_1 = np.zeros([len(self.t_c), len(self.knots)-2])
        
        for i in range(b_spline_1.shape[0]):
            for j in range(b_spline_1.shape[1]):
                basis_1 = (self.t_c[i] - self.knots[j]) / (self.knots[j+1] - self.knots[j]) * b_spline_0[i,j]
                basis_2 = (self.knots[j+2] - self.t_c[i]) / (self.knots[j+2] - self.knots[j+1]) * b_spline_0[i,j+1]
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_1[i,j] = basis_1 + basis_2
        
        b_spline_2 = np.zeros([len(self.t_c), len(self.knots)-3])
        
        for i in range(b_spline_2.shape[0]):
            for j in range(b_spline_2.shape[1]):
                basis_1 = (self.t_c[i] - self.knots[j]) / (self.knots[j+2] - self.knots[j]) * b_spline_1[i,j]
                basis_2 = (self.knots[j+3] - self.t_c[i]) / (self.knots[j+3] - self.knots[j+1]) * b_spline_1[i,j+1]
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_2[i,j] = basis_1 + basis_2
        
                
        b_spline_3 = np.zeros([len(self.t_c), len(self.knots)-4])
        
        for i in range(b_spline_3.shape[0]):
            for j in range(b_spline_3.shape[1]):
                basis_1 = (self.t_c[i] - self.knots[j]) / (self.knots[j+3] - self.knots[j]) * b_spline_2[i,j]
                basis_2 = (self.knots[j+4] - self.t_c[i]) / (self.knots[j+4] - self.knots[j+1]) * b_spline_2[i,j+1]
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_3[i,j] = basis_1 + basis_2
        
        b_spline_3_dt = np.zeros([len(self.t_c), len(self.knots)-4])
        
        for i in range(b_spline_3_dt.shape[0]):
            for j in range(b_spline_3_dt.shape[1]):
                basis_1 = b_spline_2[i,j] / (self.knots[j+3] - self.knots[j])
                basis_2 = b_spline_2[i,j+1] / (self.knots[j+4] - self.knots[j+1])
                if np.isnan(basis_1): 
                    basis_1 = 0.0
                if np.isnan(basis_2): 
                    basis_2 = 0.0
                b_spline_3_dt[i,j] = 3 * (basis_1 - basis_2)
                
        
        return b_spline_3, b_spline_3_dt

