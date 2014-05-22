import sys, os
home_dir = os.getenv("HOME")
sys.path.append(home_dir + "/virtualenv/v3/lib/python2.7/site-packages/")

import csv
import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Resampler(object):
   
    def __init__(self):
        self.data = []
        self.goods = []
        self.bads = []
        self.part = []
        self.npart = []
        self.part_rules = []
        self.npart_rules = []
        self.outliers = []
        self.trials = []
        self.rand_diffs = []
        self.hist = []
        self.sigma = 0
        self.trials_signal = 0
        self.trials_noise = 0
        self.rand_odds = 0.50
        self.rand_choices = [0,0,0,0,1] # 0 is non-part
        self.rand_groups = [1,1] # [num_parts, num_nparts] ... sum < len(goods)
        self.archive = []

    def read_data(self, csv_file, id_col):
        self.id_col = id_col
        with open(csv_file, 'rb') as csvfile: 
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                self.data.append(row)
            self.headers = self.data[0]
            del self.data[0]
        self.goods = self.data[:]

    def calc_sigma(self):
        num_splits = len(self.goods) * 10
        self.rand_diffs = []
        for ii in range(0,num_splits):
            self.rand_diffs.append(self.random_subset())
        self.sigma = np.std(self.rand_diffs)
            
    def clean_data(self, val, cols):
        for cc in cols:
            goods = [x for x in self.goods if not self.determine(val, cc, x)]
            bads = [x for x in self.goods if self.determine(val, cc, x)]
            self.goods = goods
            for bb in bads:
                self.bads.append(bb)
    
    def determine(self, val, col, data):
        return val == data[col]

    def random_subset(self):
        # subsample and then split
        random.shuffle(self.goods)
        num_part = self.rand_groups[0]
        num_npart = self.rand_groups[1]
        part = self.goods[:num_part]
        npart = self.goods[-num_npart:]
        ret = (np.mean(map(float, zip(*part)[self.comp_col])) - np.mean(map(float, zip(*npart)[self.comp_col])))
        return ret

    def begin_partition(self):
        self.part = self.goods[:]
        self.npart = self.goods[:]

    def validate_partition(self):
        self.rand_groups = [len(self.npart), len(self.part)]
        one = set(zip(*self.part)[self.id_col])
        two = set(zip(*self.npart)[self.id_col])
        print two.intersection(one)
        print len(one)
        print len(two)

    def and_parts(self, rules):
        self.part_rules += rules
        temp = []
        for par in self.part: 
            keep = True
            for rr in rules:
                try: 
                    col_val = float(par[rr[0]]) 
                except: 
                    col_val = par[rr[0]]
                if not rr[1](col_val, rr[2]):
                    keep = False
                    break
            if keep:
                temp.append(par)
        self.part = temp

    def and_nons(self, rules):
        self.npart_rules += rules
        temp = []
        for par in self.npart: 
            keep = True
            for rr in rules:
                try: 
                    col_val = float(par[rr[0]]) 
                except: 
                    col_val = par[rr[0]]
                if not rr[1](col_val, rr[2]):
                    keep = False
                    break
            if keep:
                temp.append(par)
        self.npart = temp

    def resample_deterministic(self):
        # loop over parts and identify/save neighborhood matches
        self.part_neighbors = []
        self.part_outliers = []
        for pick in self.part:
            range_min = float(pick[self.match_col]) - self.match_param
            range_max = float(pick[self.match_col]) + self.match_param
            matches = []
            for pp in self.npart:
                if(float(pp[self.match_col])>=float(range_min)) and (float(pp[self.match_col])<=float(range_max)):
                    matches.append(pp)       
            if len(matches) == 0:
                self.part_outliers.append(pick) 
            else:
                self.part_neighbors.append([pick, matches])

        # loop over every participant and identify/save their neighborhood matches
        self.npart_neighbors = []
        self.npart_outliers = []
        for pick in self.npart:
            range_min = float(pick[self.match_col]) - self.match_param
            range_max = float(pick[self.match_col]) + self.match_param
            matches = []
            for pp in self.part:
                if(float(pp[self.match_col])>=float(range_min)) and (float(pp[self.match_col])<=float(range_max)):
                    matches.append(pp)       
            if len(matches) == 0:
                self.npart_outliers.append(pick) 
            else:
                self.npart_neighbors.append([pick, matches])
 
        # compute differences between parts and neightbors 
        comp1 = []
        neighbor_dist1 = []
        for pp in self.part_neighbors:
            avg_parts = float(pp[0][self.comp_col]) - np.mean(map(float, zip(*pp[1])[self.comp_col]))
            comp1.append(avg_parts)
            neighbor_dist1.append(len(pp[1]))
        
        # compute differences between nons and neightbors 
        comp2 = []
        neighbor_dist2 = []
        for pp in self.npart_neighbors:
            avg_nparts = float(pp[0][self.comp_col]) - np.mean(map(float, zip(*pp[1])[self.comp_col]))
            comp2.append(avg_nparts)
            neighbor_dist2.append(len(pp[1]))
       
        #self.neighbors_mean = np.mean(comp1) - np.mean(comp2) # <<< inappropriate weighting
        self.neighbors_mean = (np.mean(comp1) * len(comp1) - np.mean(comp2) * len(comp2)) / (len(comp2) + len(comp1))
        self.calc_sigma()
        # archive the data
        aa = Archive()
        aa.comp_col = self.headers[self.comp_col]
        aa.num_parts = len(self.part)
        aa.num_nparts = len(self.npart)
        aa.straight_diff = self.get_straight_averages()
        aa.match_col = self.headers[self.match_col]
        aa.match_param = self.match_param
        aa.part_rules = self.part_rules
        aa.npart_rules = self.npart_rules
        pres = 0
        if len(self.part_outliers) > 0:
            pres = zip(*self.part_outliers)[self.match_col]
        aa.part_outliers = pres
        nres = 0
        if len(self.npart_outliers) > 0:
            nres = zip(*self.npart_outliers)[self.match_col]
        aa.npart_outliers = nres
        aa.neighbors_mean = round(self.neighbors_mean, 3)
        aa.neighbors_mean_error = round((self.sigma / np.sqrt(len(self.part))) + (self.sigma / np.sqrt(len(self.npart))), 3)
        aa.sigma = round(self.sigma, 3)
        aa.effect = round((self.neighbors_mean / self.sigma), 3)
        self.archive.append(aa)
    
    def print_archive_txt(self):
        for aa in self.archive:
            aa.print_txt()

    def print_archive_csv(self):
        Archive.print_headers()
        for aa in self.archive:
            aa.print_csv()

    def plot_archive(self):
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        fig=plt.figure()
        # specify size of plot attributes
        fig.set_size_inches(5.1,3.4)
        myfontsize = 12
        sp = 1
        ns = len(self.archive)
        # find scale
        sigma = 1
        max_effect = 0
        for aa in self.archive:
            if aa.effect > max_effect:
                max_effect = aa.effect 
        leftY = -3*sigma
        rightY = 3*sigma + max_effect
        #titles = {'Exam_Final_Score': 'Final Exam (0-100%)', 'Final_Percent': 'Percent of Points (0-100%)', 'Final_Grade_Real_Roster': 'Grade in Class (0-4 scale)'}
        for aa in self.archive:
            # identify the subplot
            spos = int(str(ns) + str(1) + str(sp))
            ax=fig.add_subplot(spos)
            #ax.set_title(titles[aa.comp_col])
            parts_title = ""
            for rr in self.part_rules:
                parts_title += ('(' +self.headers[rr[0]] + ' ' + rr[1].__name__ + ' ' + str(rr[2]) + ') ')
            nparts_title = ""
            for rr in self.npart_rules:
                nparts_title +=  ('(' +self.headers[rr[0]] + ' ' + rr[1].__name__ + ' ' + str(rr[2]) + ') ')
            mytitle = "Matched on " + str(aa.match_col) + " +/- " + str(aa.match_param)
            mytitle += "\nParticipants: " + str(aa.num_parts) + " where " + parts_title
            mytitle += "\nNon-participants: " + str(aa.num_nparts) + " where " + nparts_title
            ax.set_title(mytitle, fontsize=myfontsize)
            ax.set_xlabel(str(aa.comp_col), fontsize=myfontsize)
            ax.set_xlim([leftY,rightY])
            ax.set_ylim([0,0.5])
            ax.yaxis.set_visible(False)
            ax.text(-2.5, 0.47, (str(round(aa.effect, 2)) + ' sigma effect'), verticalalignment='top', fontsize=myfontsize)
            ax.text(-2.5, 0.38, (str(round(aa.neighbors_mean, 2)) + ' gain'), verticalalignment='top', fontsize=myfontsize)
            mean = 0
            variance = 1
            x = np.linspace(leftY,rightY,300)
            plt.plot(x,mlab.normpdf(x,mean,sigma), linewidth=5)
            # bars to show signals
            half_width = max((aa.neighbors_mean_error/sigma/2), 0.03)
            real_left = aa.effect - half_width
            real_right = aa.effect + half_width
            naive_left = (aa.straight_diff/aa.sigma) - half_width
            naive_right = (aa.straight_diff/aa.sigma) + half_width
            l = plt.axvspan(real_left, real_right, color='g', alpha=1)
            l = plt.axvspan(naive_left, naive_right, color='r', alpha=0.25)
            # label x-axis by scaling to sigma
            ii = 8
            xtick_extra = []
            xlabel_extra = []
            while ii < (aa.neighbors_mean/aa.sigma + 2) or ii < (aa.straight_diff/aa.sigma + 2):
                xtick_extra.append(ii)
                if ii % 2 == 0:
                    xlabel_extra.append(ii*aa.sigma)
                else:
                    xlabel_extra.append('')
                ii += 1 
            xtick_extra_left = []
            xlabel_extra_left = []
            ii = -2
            while ii > (aa.neighbors_mean/aa.sigma - 2) or ii > (aa.straight_diff/aa.sigma - 2):
                xtick_extra_left.append(ii)
                if ii % 2 == 0:
                    xlabel_extra_left.append(ii*aa.sigma)
                else:
                    xlabel_extra_left.append('')
                ii -= 1 
            #ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] + [8, 9, 10, 11])
            ax.set_xticks(xtick_extra_left + [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7] + xtick_extra)
            ax.set_xticklabels(xlabel_extra_left + ['', -2*aa.sigma, '', 0, '', 2*aa.sigma, '', 4*aa.sigma, '', 6*aa.sigma, ''] + xlabel_extra, fontsize=myfontsize)

            plt.draw()
            sp += 1 

    def get_straight_averages(self):
        # average a column -- gpa example
        avgfinscorepart = np.mean(map(float, zip(*self.part)[self.comp_col]))
        avgfinscorenpart = np.mean(map(float, zip(*self.npart)[self.comp_col]))
        return round((avgfinscorepart-avgfinscorenpart), 3)
   
    def show_plots(self):
        plt.show()
 
class Archive(object):

    def __init__(self):
        self.comp_col = 'none'
        self.num_parts = 0
        self.num_nparts = 0
        self.straight_diff = 0
        self.match_col = 'none'
        self.match_param = 0.1
        self.part_outliers = 'none'
        self.npart_outliers = 'none'
        self.part_rules = 'none'
        self.npart_rules = 'none'
        self.neighbors_mean = 0
        self.neighbors_mean_error = 0
        self.sigma = 0
        self.effect = 0

    def print_txt(self):
        print self.comp_col
        print "parts =" , str(self.num_parts), ',', "non =", str(self.num_nparts)
        print "Straight diff: " + str(self.straight_diff)
        print "Matched on " + self.match_col + " +/- " + str(self.match_param)
        print "Unmatched Part, Non:", str(self.part_outliers), ',', str(self.npart_outliers)
        print "Adjusted diff: " + str(self.neighbors_mean), '+/-', round(self.neighbors_mean_error)
        print "Sigma:", self.sigma, "Effect:", str(self.effect), 'sigma'
        print ""

    @classmethod
    def print_headers(self):
        header = ""
        header += "comparison_property,"
        header += "number_participants,"
        header += "number_non_participants,"
        header += "straight_difference,"
        header += "matching_property,"
        header += "matching_range,"
        header += "adjusted_difference,"
        header += "measurement_error,"
        header += "sigma_noise,"
        header += "size_effect"
        print header

    def print_csv(self):
        row = "" 
        row += self.comp_col + ','
        row += str(self.num_parts) + ','
        row += str(self.num_nparts) + ','
        row += str(self.straight_diff) + ','
        row += self.match_col  + ','
        row += str(self.match_param) + ','
        row += str(self.neighbors_mean) + ','
        row += str(self.neighbors_mean_error) + ','
        row += str(self.sigma) + ','
        row += str(self.effect)
        print row

