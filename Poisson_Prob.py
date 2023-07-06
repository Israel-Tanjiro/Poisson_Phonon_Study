#Class used to generate Density plot function and Poisson Analysis
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import numpy as np
from numpy import arange
from scipy.optimize import curve_fit
from matplotlib import pyplot
from scipy.stats import poisson
import random
class Poisson_Prob:
    Radius=[]
    Z_1=[]
    Z_2=[]
    Z_3=[]
    Z_4=[]
    Z_5=[]
    Z_6=[]
    Z_7=[]
    Z_8=[]
    Z_9=[]
    Z_10=[]
    Mean_Nb_01_Cu_0=[]
    Sigma_Nb_01_Cu_0=[]
    y_line_log_1_lineal=[]
    Radius_Not_1=[]
    Sigma_Nb_1_Cu_1_log_lineal=[]
    X_Qubit_0=0
    Y_Qubit_0=0
    X_Qubit_1=0
    Y_Qubit_1=0
    X_Qubit_2=0
    Y_Qubit_2=0

    X_Qubit_3=0
    Y_Qubit_3=0
    X_Qubit_4=0
    Y_Qubit_4=0
    X_Qubit_5=0
    Y_Qubit_5=0
    X_photon=0
    Y_photon=0

    L=0

    L2=0
    L3=0
    L4=0
    L5=0
    L6=0
    X_maxX=0
    def __init__(self,nameFile):
        self.nameFile= nameFile
    def Ploting_Mean(self):
        with open(self.nameFile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            list_of_column_names = []
            for row in csv_reader:
                list_of_column_names.append(row)

                        # breaking the loop after the
                        # first iteration itself
                break


            for lines in csv_reader:
                            #print(lines[9])
                        #i=1+i
                        #print(i)
                            #Energy_Initial_e.append(float(lines[4]))


                self.Radius.append(float(lines[0]))#Converting to cm from mete
                self.Z_1.append(float(lines[1]))#Converting to cm from mete
                self.Z_2.append(float(lines[2]))#Converting to cm from mete
                self.Z_3.append(float(lines[3]))#Converting to cm from mete
                self.Z_4.append(float(lines[4]))#Converting to cm from mete
                self.Z_5.append(float(lines[5]))#Converting to cm from mete
                self.Z_6.append(float(lines[6]))#Converting to cm from mete
                self.Z_7.append(float(lines[7]))#Converting to cm from mete
                self.Z_8.append(float(lines[8]))#Converting to cm from mete
                self.Z_9.append(float(lines[9]))#Converting to cm from mete
                self.Z_10.append(float(lines[10]))#Converting to cm from mete



            for i in range(0,len(self.Z_1)):
                self.Mean_Nb_01_Cu_0.append((+self.Z_1[i]+self.Z_2[i]+self.Z_3[i]+self.Z_4[i]+self.Z_5[i]+self.Z_6[i]+self.Z_7[i]+self.Z_8[i]+self.Z_9[i]+self.Z_10[i])/10.0)
                #print(self.Z_1[i])

            for i in range(0,len(self.Z_1)):
                self.Sigma_Nb_01_Cu_0.append(math.sqrt( (pow(self.Mean_Nb_01_Cu_0[i]-self.Z_1[i],2)+pow(self.Mean_Nb_01_Cu_0[i]-self.Z_2[i],2)+pow(self.Mean_Nb_01_Cu_0[i]-self.Z_3[i],2)
                +pow(self.Mean_Nb_01_Cu_0[i]-self.Z_4[i],2)+pow(self.Mean_Nb_01_Cu_0[i]-self.Z_5[i],2)
                +pow(self.Mean_Nb_01_Cu_0[i]-self.Z_6[i],2)+pow(self.Mean_Nb_01_Cu_0[i]-self.Z_7[i],2)+pow(self.Mean_Nb_01_Cu_0[i]-self.Z_8[i],2)
                +pow(self.Mean_Nb_01_Cu_0[i]-self.Z_9[i],2)+pow(self.Mean_Nb_01_Cu_0[i]-self.Z_10[i],2) ))/10.0)
# This Function is going to Fit to the mean of all files to a Polynomial

    def Fitting_to_Mean(self):

        for i in range(0,len(self.Z_1)):

            if self.Mean_Nb_01_Cu_0[i]!=0:

                if self.Radius[i]<0.7:
                    self.y_line_log_1_lineal.append(math.log10(self.Mean_Nb_01_Cu_0[i]))
                    self.Radius_Not_1.append(self.Radius[i])
                    self.Sigma_Nb_1_Cu_1_log_lineal.append(self.Sigma_Nb_01_Cu_0[i]/(2.302585093*self.Mean_Nb_01_Cu_0[i]))
        #-------------------------Doing a Fit to the exponential-----------------
        # A = 6  # Want figure to be A6
        #
        # plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # plt.rc('text.latex', preamble=r'\usepackage{textgreek}')
        # ax = plt.axes()
        # ax = plt.gca()
        #ax.errorbar(Radius_Not_1,y_line_log_1_lineal, Sigma_Nb_1_Cu_1_log_lineal,fmt='.', linewidth=2, capsize=3)
        model5 = np.poly1d(np.polyfit(self.Radius_Not_1,self.y_line_log_1_lineal,5))
        # polyline = np.linspace(0.0,0.7, 50)
        # R2=[]
        # for i in range(1,6):
        #     R2.append(round(adjR(Radius_Not_1,y_line_log_1_lineal, i),2))
        # #plt.plot(polyline, model5(polyline),'--' ,color='black',label='Pol 5    $R^{2}$='+str(R2[4]))
        # ax.xaxis.label.set_fontsize(20)
        # ax.yaxis.label.set_fontsize(20)
        # ax.get_xaxis().get_major_formatter().set_scientific(False)
        # plt.ylabel(' Mean Radial  Probability log10 ')
        # plt.xlabel('Distance to Qubit Cross [cm]')
        # # plt.yscale('logit')
        # ax.legend(fontsize='20')
        #
        # plt.xticks(fontsize = 20)
        #
        # plt.yticks(fontsize = 20)

        #plt.show()
        return model5
    def fit_function(self,k, lamb):
        # The parameter lamb will be used as the fit parameter
        return poisson.pmf(k, lamb)
    def Posion_Random(self,X,trigger):
        # i=0
        Qubit_1_random_Phonon=[]


        for i in range(0,len(X)):
            if X[i]>trigger:
                Qubit_1_random_Phonon.append(i)
        return Qubit_1_random_Phonon



    def Coincidence(self,X1,X2,X3,X4,X5,X6,Number_Phonons):
        #Counting  How Many times is Number of Phonons for all the Qubits.
        j=0
        Coincidences=[]
        for i in range(0,len(X1)):
            if X1[i]==Number_Phonons:
                j=j+1
            if X2[i]==Number_Phonons:
                j=j+1
            if X3[i]==Number_Phonons:
                j=j+1
            if X4[i]==Number_Phonons:
                j=j+1
            if X5[i]==Number_Phonons:
                j=j+1
            if X6[i]==Number_Phonons:
                j=j+1
            Coincidences.append(j)
            j=0


        return Coincidences
    def Initial_Photon(self,X,Y):
        fig,ax1 = plt.subplots()

        #plt.style.use('seaborn-pastel')
        self.X_photon =X
        self.Y_photon =Y

        # X_photon =0.8*np.random.random_sample(size=1)-0.4
        # X_MEMS=np.arange(-0.4,0.4, 0.01)
        # Y_MEMS=0.3
        # Y_photon =0.8*np.random.random_sample(size=1)-0.4
        ax1.set_xlim(-0.4,0.4)
        ax1.set_ylim(-0.4,0.4)
        x_plot = np.arange(0, 10)


        self.X_Qubit_0=self.X_photon+0.1855
        self.Y_Qubit_0=self.Y_photon-0.1772
        self.X_Qubit_1=self.X_photon+0.0
        self.Y_Qubit_1=self.Y_photon-0.1772
        self.X_Qubit_2=self.X_photon-0.1855
        self.Y_Qubit_2=self.Y_photon-0.1772

        self.X_Qubit_3=self.X_photon+0.1855
        self.Y_Qubit_3=self.Y_photon+0.1772
        self.X_Qubit_4=self.X_photon+0.0
        self.Y_Qubit_4=self.Y_photon+0.1772
        self.X_Qubit_5=self.X_photon-0.1855
        self.Y_Qubit_5=self.Y_photon+0.1772
        ax1.plot(self.X_photon, self.Y_photon, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="red")

        ax1.plot(-0.1855, 0.1772, marker="+", markersize=30, markeredgecolor="blue", markerfacecolor="blue")
        ax1.plot(0.0, 0.1772, marker="+", markersize=30, markeredgecolor="blue", markerfacecolor="blue")
        ax1.plot(0.1855, 0.1772, marker="+", markersize=30, markeredgecolor="blue", markerfacecolor="blue")
        ax1.plot(-0.1855, -0.1772, marker="+", markersize=30, markeredgecolor="blue", markerfacecolor="blue")
        ax1.plot(0.0,- 0.1772, marker="+", markersize=30, markeredgecolor="blue", markerfacecolor="blue")
        ax1.plot(0.1855, -0.1772, marker="+", markersize=30, markeredgecolor="blue", markerfacecolor="blue")
        ax1.set_xlabel('Photon Starting X cm')
        ax1.set_ylabel('Photon Starting Y cm')
        Xii=round(self.X_photon,2)
        Yii=round(self.Y_photon,2)
        ax1.text(0.0 ,0.3,'X={}'.format(Xii))
        ax1.text(0.1 ,0.3,'Y={}'.format(Yii))
        plt.show()


    def Poisson_Plots(self,Phonon_Energy):
        R_randon_1=np.sqrt(self.X_Qubit_0*self.X_Qubit_0+self.Y_Qubit_0*self.Y_Qubit_0)
        R_randon_2=np.sqrt(self.X_Qubit_1*self.X_Qubit_1+self.Y_Qubit_1*self.Y_Qubit_1)
        R_randon_3=np.sqrt(self.X_Qubit_2*self.X_Qubit_2+self.Y_Qubit_2*self.Y_Qubit_2)
        R_randon_4=np.sqrt(self.X_Qubit_3*self.X_Qubit_3+self.Y_Qubit_3*self.Y_Qubit_3)
        R_randon_5=np.sqrt(self.X_Qubit_4*self.X_Qubit_4+self.Y_Qubit_4*self.Y_Qubit_4)
        R_randon_6=np.sqrt(self.X_Qubit_5*self.X_Qubit_5+self.Y_Qubit_5*self.Y_Qubit_5)
        #Call the Function of the Object
        PDF=self.Fitting_to_Mean()
        Phonons_Number=int(Phonon_Energy/0.005)
        Phonons_Distribution_Qubit_1=Phonons_Number*np.power(10,PDF(R_randon_1))
        Phonons_Distribution_Qubit_2=Phonons_Number*np.power(10,PDF(R_randon_2))
        Phonons_Distribution_Qubit_3=Phonons_Number*np.power(10,PDF(R_randon_3))
        Phonons_Distribution_Qubit_4=Phonons_Number*np.power(10,PDF(R_randon_4))
        Phonons_Distribution_Qubit_5=Phonons_Number*np.power(10,PDF(R_randon_5))
        Phonons_Distribution_Qubit_6=Phonons_Number*np.power(10,PDF(R_randon_6))


        self.L=Phonons_Distribution_Qubit_1
        self.L2=Phonons_Distribution_Qubit_2
        self.L3=Phonons_Distribution_Qubit_3
        self.L4=Phonons_Distribution_Qubit_4
        self.L5=Phonons_Distribution_Qubit_5
        self.L6=Phonons_Distribution_Qubit_6

        X_maxx=[]
        X_maxx.append(int(self.L))
        X_maxx.append(int(self.L2))
        X_maxx.append(int(self.L3))
        X_maxx.append(int(self.L4))
        X_maxx.append(int(self.L5))
        X_maxx.append(int(self.L6))
        self.X_maxX=np.max(X_maxx)

        x_plot = np.arange(0,self.X_maxX+11)

        t=plt.bar(x_plot,self.fit_function(x_plot,self.L ),width=0.1,align='center')
        t2=plt.bar(x_plot+0.1,self.fit_function(x_plot,self.L2 ),width=0.1)
        t3=plt.bar(x_plot+0.2,self.fit_function(x_plot,self.L3 ),width=0.1,align='center')
        t4=plt.bar(x_plot+0.3,self.fit_function(x_plot,self.L4 ),width=0.1)
        t5=plt.bar(x_plot+0.4,self.fit_function(x_plot,self.L5 ),width=0.1,align='center')
        t6=plt.bar(x_plot+0.5,self.fit_function(x_plot,self.L6 ),width=0.1)
        plt.legend([t,t2,t3,t4,t5,t6],['Qubit 1 ','Qubit 2','Qubit 3 ','Qubit 4','Qubit 5 ','Qubit 6'])
        #plt.gca().add_artirst(first_legend)
        bin=[]
        for i in range(0,self.X_maxX+11):
            bin.append(i)

        plt.xticks(np.arange(min(bin), max(bin)+1, 1.0))
        # plt.gca().annotate('X={}'.format(Xii), [2, 0.9])
        # plt.gca().annotate('Y={}'.format(Yii), [2.8, 0.9])
        plt.gca().set_xlabel('Number of Phonons absorbed on Qubits',fontsize=20)
        plt.gca().set_ylabel('Probability',fontsize=20)

        plt.yscale("log")
        # plt.text(2, 0.8, r'X=%s,Y=%s', fontsize=15)
        plt.show()
    def Qubits_Aborbed_phonons(self,trigger,Phonons_Absorbed):
        NPA=str(Phonons_Absorbed)
        PAH=Phonons_Absorbed
        Xtitle='Number of Qubits observating '
        Pu=' Phonons '
        Xtitle=Xtitle+NPA
        Xtitle=Xtitle+Pu
        labelo=Pu+NPA

        x_plot = np.arange(0,self.X_maxX+11)
        Qubit_1_Possion_Values=self.fit_function(x_plot,self.L)
        Qubit_2_Possion_Values=self.fit_function(x_plot,self.L2)
        Qubit_3_Possion_Values=self.fit_function(x_plot,self.L3)
        Qubit_4_Possion_Values=self.fit_function(x_plot,self.L4)
        Qubit_5_Possion_Values=self.fit_function(x_plot,self.L5)
        Qubit_6_Possion_Values=self.fit_function(x_plot,self.L6)


        #Selecting Values less than some probability
        Prob_tigger=trigger
        Q1=self.Posion_Random(Qubit_1_Possion_Values,Prob_tigger)
        #print(Q1)
        Q2=self.Posion_Random(Qubit_2_Possion_Values,Prob_tigger)
        Q3=self.Posion_Random(Qubit_3_Possion_Values,Prob_tigger)
        Q4=self.Posion_Random(Qubit_4_Possion_Values,Prob_tigger)
        Q5=self.Posion_Random(Qubit_5_Possion_Values,Prob_tigger)
        Q6=self.Posion_Random(Qubit_6_Possion_Values,Prob_tigger)
        #Doing the Random test 1000 times and see how the Distribution looks like
        Random_Phonon_Choice_Qubit1=[]
        Random_Phonon_Choice_Qubit2=[]
        Random_Phonon_Choice_Qubit3=[]
        Random_Phonon_Choice_Qubit4=[]
        Random_Phonon_Choice_Qubit5=[]
        Random_Phonon_Choice_Qubit6=[]


        for i in range (0,1000):
            Random_Phonon_Choice_Qubit1.append(random.choice(Q1))
            Random_Phonon_Choice_Qubit2.append(random.choice(Q2))
            Random_Phonon_Choice_Qubit3.append(random.choice(Q3))
            Random_Phonon_Choice_Qubit4.append(random.choice(Q4))
            Random_Phonon_Choice_Qubit5.append(random.choice(Q5))
            Random_Phonon_Choice_Qubit6.append(random.choice(Q6))
        #print(Random_Phonon_Choice_Qubit1)
        Q_C1=self.Coincidence(Random_Phonon_Choice_Qubit1,Random_Phonon_Choice_Qubit2,Random_Phonon_Choice_Qubit3,Random_Phonon_Choice_Qubit4,Random_Phonon_Choice_Qubit5,Random_Phonon_Choice_Qubit6,PAH)
        # Q_C2=self.Coincidence(Random_Phonon_Choice_Qubit1,Random_Phonon_Choice_Qubit2,Random_Phonon_Choice_Qubit3,Random_Phonon_Choice_Qubit4,Random_Phonon_Choice_Qubit5,Random_Phonon_Choice_Qubit6,1)
        # Q_C3=self.Coincidence(Random_Phonon_Choice_Qubit1,Random_Phonon_Choice_Qubit2,Random_Phonon_Choice_Qubit3,Random_Phonon_Choice_Qubit4,Random_Phonon_Choice_Qubit5,Random_Phonon_Choice_Qubit6,2)

        plt.hist(Q_C1,bins=[0,1,2,3,4,5,6,7],color='blue',width=0.2 ,density=True, label=labelo)
        plt.gca().set_xlabel(Xtitle,fontsize=20)
        plt.gca().set_ylabel('Counts',fontsize=20)
        # plt.hist(Coincidence(Random_Phonon_Choice_Qubit1,Random_Phonon_Choice_Qubit2,Random_Phonon_Choice_Qubit3,Random_Phonon_Choice_Qubit4,Random_Phonon_Choice_Qubit5,Random_Phonon_Choice_Qubit6,3)
        # ,bins='auto',color='pink' ,density=False, label='Data')
        plt.legend()
        plt.show()

    def Mean_Number_of_Qubit_by_Hit(self, Sample):
        Nn=Sample
        #Here Plotting the Histograms Using the Mean.
        s = np.random.poisson(self.L, Nn)
        s1 = np.random.poisson(self.L2, Nn)
        s2 = np.random.poisson(self.L3, Nn)
        s3 = np.random.poisson(self.L4, Nn)
        s4 = np.random.poisson(self.L5, Nn)
        s5 = np.random.poisson(self.L6, Nn)
        #Counting the mean of every s.

        #print(s)
        #count, bins, ignored = plt.hist(s, 14, density=True)
        plt.show()
        Xxx='Qubit_1,Qubit_2,Qubit_3,Qubit_4,Qubit_5,Qubit_6'
        Yyy=[np.mean(s),np.mean(s1),np.mean(s2),np.mean(s3),np.mean(s4),np.mean(s5)]
        Sigma_Yyy=[np.std(s)/np.sqrt(Nn),np.std(s1)/np.sqrt(Nn),np.std(s2)/np.sqrt(Nn),np.std(s3)/np.sqrt(Nn),np.std(s4)/np.sqrt(Nn),np.std(s5)/np.sqrt(Nn)]
        print(Yyy)
        plt.bar('Qubit_1',Yyy[0],yerr=Sigma_Yyy[0],width=0.7,align='center',ecolor='black',capsize=5)
        plt.bar('Qubit_2',Yyy[1],yerr=Sigma_Yyy[1],width=0.7,align='center',ecolor='black',capsize=5)
        plt.bar('Qubit_3',Yyy[2],yerr=Sigma_Yyy[2],width=0.7,align='center',ecolor='black',capsize=5)
        plt.bar('Qubit_4',Yyy[3],yerr=Sigma_Yyy[3],width=0.7,align='center',ecolor='black',capsize=5)
        plt.bar('Qubit_5',Yyy[4],yerr=Sigma_Yyy[4],width=0.7,align='center',ecolor='black',capsize=5)
        plt.bar('Qubit_6',Yyy[5],yerr=Sigma_Yyy[5],width=0.7,align='center',ecolor='black',capsize=5)
        plt.gca().set_ylabel('average # of hits per event',fontsize=20)

        plt.yscale("log")
        plt.show()

    def Cleaner(self):

        self.Radius.clear()
        self.Z_1.clear()
        self.Z_2.clear()
        self.Z_3.clear()
        self.Z_4.clear()
        self.Z_5.clear()
        self.Z_6.clear()
        self.Z_7.clear()
        self.Z_8.clear()
        self.Z_9.clear()
        self.Z_10.clear()
        self.Mean_Nb_01_Cu_0.clear()
        self.Sigma_Nb_01_Cu_0.clear()
