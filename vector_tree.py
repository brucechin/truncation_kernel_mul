import numpy as np 

import argparse
import matplotlib.pyplot as plt
import time

def rand_gen_vector(size):
    a = np.random.randint(1000000, size = size) - 500000
    # a = np.random.randint(1000000, size = size)/1000000.0
    # histogram = np.zeros(100)
    # for i in range(size):
    #     index = (int) (a[i]/0.01)
    #     histogram[index]+=1  
    # print(histogram)
    return a/50000.0


def vector_diff_norm(x, y):
    return np.linalg.norm(x - y, ord = 2)


def vector_error_l1(ours, correct):
    # print(ours, correct)
    res = np.linalg.norm(ours - correct, ord = 1) / np.linalg.norm(correct, ord = 1)
    # res2 = 0
    # correct_base = 0
    # for i in range(len(ours)):
    #     res2 += abs(ours[i] - correct[i])
    #     correct_base += abs(correct[i])
    # print(res, res2/correct_base)
    return res

def vector_error_l2(ours, correct):
    return np.sqrt(np.linalg.norm(ours - correct, ord = 2) / np.linalg.norm(correct, ord = 2))

# we use vector of vectors to represent the tree. note the simulation performance will be affected by the hardware events like cache hit/miss
def makeMaxTree(input):
    num_leaf_nodes = len(input)
    height = np.log2(len(input)).astype(int) + 1
    res = [[] for i in range(height)]
    res[height - 1] = input 
    cur_level = height - 1 
    while(cur_level > 0):
        tmp = []
        size = 2 ** (cur_level - 1)
        for i in range(size):
            tmp.append(max(res[cur_level][2 * i], res[cur_level][2 * i + 1]))
        cur_level-=1
        res[cur_level] = tmp
    return res

def query(tree, threshold):
    height = len(tree)
    indexes = [0]
    cur_level = 0
    # print(height)
    # for i in range(height):
    #     print(len(tree[i]))
    while(cur_level < height - 1):
        tmp = []
        for index in indexes:
            if(tree[cur_level +1][2 * index] >= threshold):
                tmp.append(2 * index)
            if(tree[cur_level +1][2 * index + 1] >= threshold):
                tmp.append(2 * index + 1)
        indexes = tmp
        cur_level += 1
    return indexes

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('kernel', metavar='N', type=int, nargs='+',
                    help='choice of kernel')

args = parser.parse_args()

n  = 2048
m = 2048
d = 64

# repeat 100 times to obtain the average time consumption and accuracy.
repeat = 100


X = []
Y = []
K = np.zeros((n, m))
truncation_condition = np.zeros((n, m))




def multiply(alltrees, v, threshold, flag):
    res = np.zeros(n)
    counter = 0
    for i in range(n):
        indexes = query(alltrees[i], threshold)
        for j in indexes:
            res[i] += K[i][j] * v[j]
            counter +=1.0
    return res


def multiply_baseline(v, threshold, flag):
    res = np.zeros(n)
    counter = 0
    for i in range(n):
        for j in range(m):
            if(truncation_condition[i][j] >= threshold):
                res[i] += K[i][j] * v[j]
                counter +=1.0
    # if(flag):
        # print("truncation degree={}".format(counter/(n*m)))
    return res


def generate_far_away_vectors(n, d):
    distance_threshold = d * 0.1
    results = []
    while(len(results) < n):
        tmp = rand_gen_vector(d)
        real_far_away_to_all = True
        for r in results:
            dis = np.sqrt(np.linalg.norm(tmp - r, ord = 2))
            if(dis < distance_threshold):
                real_far_away_to_all = False
                break
        if(real_far_away_to_all):
            results.append(tmp)
    return results

def generate_cluster_points(point, cluster_size):
    cluster = []
    # print(cluster_size)
    for i in range(cluster_size - 1):
        tmp = []
        # print(len(point))
        for j in range(len(point)):
            tmp.append(point[j] + np.random.random_sample() * 0.01)
        # print(vector_diff_norm(tmp, point))
        # print(tmp, point)
        cluster.append(np.array(tmp))
    return cluster

#first generate n points which are far away from each other

k = 32
cluster_size = (int)(n / k)

dis_x = generate_far_away_vectors(k, d)
X = []
for i in range(k):
    # then generate k clustered points(very near within each cluster)
    cluster = generate_cluster_points(dis_x[i], cluster_size)
    X.append(dis_x[i])
    for ele in cluster:
        X.append(ele)
# print("finish x generation")

Y = X





kernel_string = ""

kernel_selection = args.kernel[0]
for i in range(n):
    for j in range(m):
        vec_diff_norm = vector_diff_norm(X[i], Y[j])
        if(kernel_selection == 0):
            K[i][j] = np.exp(np.dot(X[i], Y[j])) 
            kernel_string = "exp"
        elif(kernel_selection == 1):
            K[i][j] = np.exp(- abs(vec_diff_norm)) 
            kernel_string ="laplacian"
        elif(kernel_selection == 2):
            K[i][j] = np.exp( - abs(vec_diff_norm) * abs(vec_diff_norm)) 
            kernel_string="gaussian"
        elif(kernel_selection == 3):
            K[i][j] = 1.0/ (1.0 + pow(vec_diff_norm, 4)) 
            kernel_string="tstudent"
        elif(kernel_selection ==4):
            K[i][j] = np.tanh(np.dot(X[i], Y[j])) 
            kernel_string="tanh"
        elif(kernel_selection ==5):
            K[i][j] = pow(np.dot(X[i], Y[j]), 4) 
            kernel_string="polynomial"
        elif(kernel_selection == 6):
            K[i][j] = 1.0/(1.0 + vec_diff_norm**2) 
            kernel_string="rational"
        elif(kernel_selection == 7):
            K[i][j] = np.sqrt(vec_diff_norm **2 + 1.0) 
            kernel_string="multiquadratic"
        elif(kernel_selection == 8):
            K[i][j] = 1.0/(np.sqrt(vec_diff_norm **2 + 1.0)) 
            kernel_string="inverse_multiquadratic"
        elif(kernel_selection == 9):
            K[i][j] = - vec_diff_norm * vec_diff_norm * vec_diff_norm * vec_diff_norm 
            kernel_string="power"
        else:
            K[i][j] = - np.log(pow(vec_diff_norm, 4) + 1)
            kernel_string= "log"
        if(vec_diff_norm == 0):
            K[i][j] = 0 #means they are the same point, we should set as 0
        truncation_condition[i][j] = K[i][j] # np.dot(X[i], Y[j])

allTrees = []
for i in range(n):
    allTrees.append(makeMaxTree(K[i]))




#1000 slices to draw the function CDF and PDF figures
Z = np.concatenate(K)
# method 1
H, X1 = np.histogram(Z, bins = 1000, normed = True )
dx = X1[1] - X1[0]
F1 = np.cumsum(H)*dx

plt.figure()
# plt.title("{} kernel value CDF".format(kernel_string), fontsize=22)
plt.plot(X1[1:], F1)
plt.xticks(fontsize = 18, rotation= 25)
plt.yticks(fontsize = 18)
plt.xlabel("Value", fontsize=22)
plt.ylabel("CDF percentage", fontsize=22)
plt.savefig("kernel_cdf_{}.pdf".format(kernel_string), bbox_inches="tight")



H, X1 = np.histogram(Z, bins = 1000, normed = True )
dx = X1[1] - X1[0]
plt.figure()
# plt.title("{} kernel value PDF".format(kernel_string), fontsize=22)
y = H * dx
plt.plot(X1[1:].round(2), y)
#comment out below line if we want the xticks
# plt.xticks([])
plt.xticks(fontsize = 18, rotation= 25)
plt.yticks(fontsize = 18)
plt.xlabel("Value", fontsize=22)
plt.ylabel("PDF percentage", fontsize=22)
plt.savefig("kernel_pdf_{}.pdf".format(kernel_string), bbox_inches="tight")

Z_size = m * n
sorted_Z = sorted(Z)
# print(sorted_Z)
# we need the truncated percentage to be small so that the computation can be accelerated.
percentage_list = np.linspace(0.9695, 0.9745, 10)

error_l1_list = []
error_l2_list = []
ours_time = []
baseline_time = []


for k in range(len(percentage_list)):
    error_l1 = 0 
    error_l2 = 0 
    threshold = sorted_Z[(int)(Z_size * percentage_list[k])]
    # below code is for l1 and l2 error calculation

    # # print('threshold=', threshold)
    # for i in range(repeat):
    #     v = rand_gen_vector(m)
    #     res = multiply_baseline(v, threshold, i == 0)
    #     true_res = np.dot(K, v)
    #     error_l1 += vector_error_l1(res, true_res)
    #     error_l2 += vector_error_l2(res, true_res)
    # # print('error=', error_l1/repeat, error_l2/repeat)
    # error_l1_list.append(error_l1/repeat)
    # error_l2_list.append(error_l2/repeat)
    # # print("error_rate_l1_{}={}".format(kernel_string, error_l1_list))
    # # print("error_rate_l2_{}={}".format(kernel_string, error_l2_list))

    #below code benchmark the time speedup
    start = time.time()
    for i in range(repeat):
        v = rand_gen_vector(m)
        res = multiply(allTrees, v, threshold, i == 0)
    end = time.time()
    ours_time.append(end - start)
    start = time.time()
    for i in range(repeat ):
        v = rand_gen_vector(m)
        res = multiply_baseline(v, threshold, i == 0)
    end = time.time()
    baseline_time.append(end - start)

print("error_rate_l1_{}={}".format(kernel_string, error_l1_list))
print("error_rate_l2_{}={}".format(kernel_string, error_l2_list))

print("time_ours_{}={}".format(kernel_string, ours_time))
print("time_baseline_{}={}".format(kernel_string, baseline_time))