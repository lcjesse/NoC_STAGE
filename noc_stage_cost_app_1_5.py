0##########Try to optimize latency of mesh###################
#################################### Import ###############################
from collections import defaultdict
from heapq import *
import xlrd
from copy import deepcopy
from random import randint,random,seed
import math
import numpy
import random
from sklearn.ensemble import RandomForestRegressor
import timeit

################################ defn. Variables #############################

'''
defn. of last move and best move:
last_move[0]: core = 0, link = 1
last_move[1]: swapped core index 1
last_move[2]: swapped core index 2
last_move[3]: removed link source
last_move[4]: removed link destination
last_move[5]: added link source
last_move[6]: added link destination
'''
#################### Creating nodes ##########################
def create_nodes(num_nodes):
    nodes = []
    hqm = 0
    cpu = 1
    pe = 5
    ddr = 10
    # for i in range(0, num_nodes):
    #     nodes.append(i)         ## if we use basic benchmark, there are no pe or ddr, just cores
    for i in range(0, num_nodes):
       nodes.append(9999)
    # hqm is 0
    # cpu are 1-4
    # pe are 5-9
    # ddr are 10-11 
    nodes[0] = hqm      #0
    nodes[1] = cpu      #1
    nodes[2] = pe       #5
    nodes[3] = ddr      #10
    nodes[4] = pe + 1   #6
    nodes[5] = pe + 2   #7
    nodes[6] = cpu + 1  #2
    nodes[7] = cpu + 2  #3
    nodes[8] = pe + 3   #8
    nodes[9] = ddr + 1  #11
    nodes[10] = pe + 4  #9
    nodes[11] = cpu + 3 #4

    return deepcopy(nodes)

########################## Creating row*column mesh link connectivity #############################
def create_mesh(row, col):
    num_cores = row * col
    links = numpy.zeros(shape=(num_cores, num_cores))
    for i in range(0, num_cores):
        for j in range(0, num_cores):
            ys = int(i / col)
            xs = i % col
            yd = int(j / col)
            xd = j % col
            if (ys == yd) and (abs(xd - xs) == 1):  # x-links
                links[i][j] = 1
            elif (xs == xd) and (abs(yd - ys) == 1):  # y-links
                links[i][j] = 1
    return deepcopy(links)
'''
################################### Make traffic ####################################
## make traffic file needed by load_traffic(), based on info in stp file
def make_traffic(traffic_file):
    with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+ traffic_file+".stp") as f:
    #with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/h263e_mesh_4x4.stp") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        ## line 15 in stp file contains number of node, task, edge
        temp = content[15].split()
        num_node = int(temp[1])
        num_task = int(temp[2])
        num_edge = int(temp[3])
        traffic = [[0 for col in range(num_node)] for row in range(num_node)]
        ## every row refers to src_node; every column refers to dst_node
        for i in range(16+num_task, 16+num_task+num_edge):
            temp1 = content[i].split()
            traffic[int(temp1[3])][int(temp1[4])] += 1
    ## write traffic into txt file
    t=''
    with open ('/content/drive/My Drive/Colaboratory/NoC_STAGE/input_traffic.txt','w') as q:
        for i in range(len(traffic)):
            for e in range(len(traffic[i])):
                t=t+str(traffic[i][e])+' '
            q.write(t.strip(' '))
            q.write('\n')
            t=''
    ## make statistics on traffic, record every node's data volume(in+out)
    node_data_volume = [0 for row in range(num_node)]
    for i in range(num_node):
        for j in range(num_node):
            node_data_volume[i] += traffic[i][j] + traffic[j][i]
'''

################################### Construct architecture ####################################
def construct_architecture(architecture_file):
    with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+architecture_file+".arch") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        num_nodes = int(content[0])      ## line 0 is number of nodes
        arch = []
        for i in range(1, 1 + num_nodes):
            temp = content[i].split()
            print(temp)
            num_cores_in_node = int(temp[1])
            t = []
            for j in range(2, 3*num_cores_in_node, 3):      ## element 2 5 8 11... is core id
                t.append(int(temp[j]))
            arch.append(t)
    return arch

################################### Change traffic file based on arch file and stp file ####################################
def make_traffic(arch, traffic_file, arch_traffic_file):
    with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+traffic_file+".stp") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        ## line 15 in stp file contains number of node, task, edge
        temp = content[15].split()
        num_node = int(temp[1])
        num_task = int(temp[2])
        num_edge = int(temp[3])
    ## write changed traffic into stp file
    t=''
    with open ("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+arch_traffic_file+".stp",'w') as q:
        for i in range(15):
            q.write(content[i])
            q.write('\n')
        temp15 = content[15].split()
        temp15[1] = str(len(arch))
        t = ''
        for e in range(len(temp15)):
            t=t+str(temp15[e])+'  '
        q.write(t)
        q.write("\n")
        for i in range(16, 16+num_task):    ## task info: change processor mapping
            temp1 = content[i].split()
            for j in range(0, len(arch)):
                for k in range(0, len(arch[j])):
                    if(int(temp1[1]) == arch[j][k]):
                        temp1[1] = j
                        break
            
            t = ''
            for e in range(len(temp1)):
                t=t+str(temp1[e])+'  '
            q.write(t)
            q.write("\n")
            t = ''
        for i in range(16+num_task, 16+num_task+num_edge):    ## edge info: change src and dst processor mapping 
            temp2 = content[i].split()
            for j in range(0, len(arch)):
                for k in range(0, len(arch[j])):
                    if(int(temp2[3]) == arch[j][k]):
                        temp2[3] = j
                        break
            for j in range(0, len(arch)):
                for k in range(0, len(arch[j])):
                    if(int(temp2[4]) == arch[j][k]):
                        temp2[4] = j
                        break
            t = ''
            for e in range(len(temp2)):
                t=t+str(temp2[e])+'  '
            q.write(t)
            q.write("\n")
            t = ''

    with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+arch_traffic_file+".stp") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        ## line 15 in stp file contains number of node, task, edge
        temp = content[15].split()
        num_node = int(temp[1])
        num_task = int(temp[2])
        num_edge = int(temp[3])
        traffic = [[0 for col in range(num_node)] for row in range(num_node)]
        ## every row refers to src_node; every column refers to dst_node
        for i in range(16+num_task, 16+num_task+num_edge):
            temp1 = content[i].split()
            traffic[int(temp1[3])][int(temp1[4])] += 1
    ## write traffic into txt file
    t=''
    with open ('/content/drive/My Drive/Colaboratory/NoC_STAGE/input_traffic.txt','w') as q:
        for i in range(len(traffic)):
            for e in range(len(traffic[i])):
                t=t+str(traffic[i][e])+' '
            q.write(t.strip(' '))
            q.write('\n')
            t=''

################################### Load traffic ####################################
def load_traffic():
    with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/input_traffic.txt") as f:
        content = f.readlines()
    # also remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    traffic = []
    injection=[]
    for i in range(0, len(content)):            ## every line is an entry of traffic, injection is the sum of this line
        temp = content[i].split()
        t = []
        injection1=0.0
        for j in range(0, len(temp)):
            t.append(float(temp[j]))
            injection1=injection1+float(temp[j])
        traffic.append(t)
        injection.append(injection1)
        
    return traffic, injection


################################# Calculate hop count #################################
def calculate_hop_count(nodes,col):
    hop_count = []
    for i in range(0, len(nodes)):
        t = []
        for j in range(0, len(nodes)):
            x1 = i%col
            x2 = j%col
            y1 = int(i/col)
            y2 = int(j/col)
            t.append(abs(x1-x2) + abs(y1-y2))      ## hop count is difference/4 + difference%4
        hop_count.append(t)
    return hop_count


################################# Calculate communication cost #################################
def calculate_cost(traffic, hop_count, nodes):
    communi_cost = 0
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            communi_cost += traffic[nodes[i]][nodes[j]] * hop_count[i][j] / 10       ## communication cost is hop*traffic volume
    return communi_cost

################################# XY routing algorithm ###############################
def xy_routing(nodes, s, d, col):        ## edges, src_node, dst_node(s and d means the place, not the content)
    x1 = s % col
    x2 = d % col
    y1 = int(s / col)
    y2 = int(d / col)
    x_diff = x2 - x1
    y_diff = y2 - y1
    path = (nodes[s])
    cost = 0
    while (x_diff != 0) or (y_diff != 0):      ## first x then y
        if x_diff < 0:
            s -= 1
            x_diff += 1
            cost += 1
            path = (nodes[s], path)
        elif x_diff > 0:
            s += 1
            x_diff -= 1
            cost += 1
            path = (nodes[s], path)
        elif y_diff < 0:
            s -= col
            y_diff += 1
            cost += 1
            path = (nodes[s], path)
        elif y_diff > 0:
            s += col
            y_diff -= 1
            cost += 1
            path = (nodes[s], path)
    return (cost, path)

################################# Dijkstra shortest path algorithm ###############################
def dijkstra(edges, f, t):      ## edges, node, node
    g = defaultdict(list)           ## g is oriented graph list
    for l,r,c in edges:
        g[l].append((c,r))          ## l means src_node, c mains distance or weight, r means dst_node

    q, seen = [(0,f,())], set()
    while q:
        (cost,v1,path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t: return (cost, path)

            for c, v2 in g.get(v1, ()):         ## find path from oriented graph
                if v2 not in seen:
                    heappush(q, (cost+c, v2, path))         ## Question: how does it find shortest path?

    return float("inf")

def make_edges(nodes,links,col):
    edges=[]
    temp=[]
    num_nodes = len(nodes)
    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            if links[i][j] == 1:
                temp.append(nodes[i])
                temp.append(nodes[j])
                ys=int(i/col)
                xs=i%col

                yd = int(j / col)
                xd = j % col
                if (xs!=xd or ys!=yd):
                    if(ys==yd):
                        temp.append(3+ math.ceil((((xd-xs)**2)+((ys-yd)**2))**0.5))
                    elif(xs==xd):
                        temp.append(3+ 2*math.ceil((((xd-xs)**2)+((ys-yd)**2))**0.5))
                    else:
                        temp.append(3+ math.ceil((((xd-xs)**2)+((ys-yd)**2))**0.5))         ## why has a base of 3 ?
                else:
                    print("link perturbation went wrong: ", i, " , ", j)
                edges.append(temp)
                #edges1.append(temp)
                temp = []
                # c=c+1
    return edges            ## edges contains three elements: src_node, dst_node, distance

####################################### params calculation ##########################
## Calculate mean and deviation of link utilization
def calc_params(nodes,links,row,col, traffic):
    m = 0
    d = 0
    link_util = []
    num_nodes = len(nodes)
    edges = make_edges(nodes, links, col)
    for i in range(0, num_nodes):
        t = []
        for j in range(0, num_nodes):
            t.append(0)
        link_util.append(t)
    # dev
    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            if nodes[i] == nodes[j]:
                continue
            # print(i, "  ", j)
            p = str(xy_routing(nodes, i, j, col))
            #p = str(dijkstra(edges, nodes[i], nodes[j]))
            p_break = p.split(',')
            for k in range(1, len(p_break) - 2):
                node1 = int(p_break[k][2:])
                node2 = int(p_break[k + 1][2:])
                ind1 = nodes.index(node1)
                ind2 = nodes.index(node2)
                if links[ind1][ind2] != 1:
                    print('something is wrong..!!')
                link_util[ind1][ind2] = link_util[ind1][ind2] + traffic[nodes[i]][nodes[j]]     ## calculate link utilization

    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            m = m + link_util[i][j]
    m = m / (2 * row * col - row - col)             ## each planar has (col-1)*row+(row-1)*col = 2*row*col-row-col links
    for i in range(0, num_nodes):
        for j in range(i, num_nodes):
            if (links[i][j] != 1):
                continue
            d = d + (link_util[i][j] + link_util[j][i] - m)**2
    d = d**0.5

    return m, d

################################ PHV calculator #####################################
##Pareto hypervolume: used to evaluate MMO 
def dominates(p, q, k=None):
    if k is None:
        k = len(p)
    d = True
    while d and k < len(p):
        d = (q[k] > p[k])
        k += 1
    return d

def insert(p, k, pl):
    ql = []
    while pl and pl[0][k] > p[k]:
        ql.append(pl[0])
        pl = pl[1:]
    ql.append(p)
    while pl:
        if not dominates(p, pl[0], k):
            ql.append(pl[0])
        pl = pl[1:]
    return ql

def slice(pl, k, ref):
    p = pl[0]
    pl = pl[1:]
    ql = []
    s = []
    while pl:
        ql = insert(p, k + 1, ql)
        p_prime = pl[0]
        s.append(((p[k] - p_prime[k]), ql))
        p = p_prime
        pl = pl[1:]
    ql = insert(p, k + 1, ql)
    s.append(((p[k] - ref[k]), ql))
    return s

def phv_calculator(archive, new_pt):
    ps = deepcopy(archive)
    ps.append(deepcopy(new_pt))
    ref = [0,0]
    n = min([len(p) for p in ps])
    pl = ps[:]
    pl.sort(key=lambda x: x[0], reverse=True)
    s = [(1, pl)]
    for k in range(n - 1):
        s_prime = []
        for x, ql in s:
            for x_prime, ql_prime in slice(ql, k, ref):
                s_prime.append((x * x_prime, ql_prime))
        s = s_prime
    vol = 0
    for x, ql in s:
        vol = vol + x * (ql[0][n - 1] - ref[n - 1])
    return vol
    #print(vol)


############################### Make Perturbation #################################

def perturb(nodes, links, case, last_move):
    num_nodes = len(nodes)
    r1 = random.random()
    if case==0:
        threshold=0.6           ## 60% change cores; 40% change links
    else:
        threshold=1.1           ## 100% change cores; 0% change links
    if (r1 < threshold):  # exchange cores 0.6
        while 1:
            rs = random.randint(0, num_nodes-1)
            rd = random.randint(0, num_nodes-1)
            i = nodes[rs]
            j = nodes[rd]           ##Why use i & j??? Not used 
            t1 = nodes[rs]
            nodes[rs] = nodes[rd]
            nodes[rd] = t1  #  exchanged nodes
            last_move[0] = 0 # core swap
            last_move[1] = rs   ## swapped core index 1
            last_move[2] = rd   ## swapped core index 2
            last_move[3] = -1
            last_move[4] = -1
            last_move[5] = -1
            last_move[6] = -1
            break
    
    else:  # change links           ## actually we DON'T use it
        while 1:
            rl1 = random.randint(0, 3)
            rs1 = random.randint(0, 15)
            rd1 = random.randint(0, 15)
            rl2 = random.randint(0, 3)
            rs2 = random.randint(0, 15)
            rd2 = random.randint(0, 15)
            if (rs1 == rd1) or (rs2 == rd2):        ## 29/225 probability
                continue
            l1 = links[rl1 * 16 + rs1][rl1 * 16 + rd1]  # remove            ## rl from 0-3 Is it means that they are in same xy plate?
            l2 = links[rl2 * 16 + rs2][rl2 * 16 + rd2]  # add               ## This work is based on 3D mesh but allow to link between distant links in each planar
            if (l1 == 0) or (l2 == 1):  # link absent/present
                continue

            # move links
            links[rl1 * 16 + rs1][rl1 * 16 + rd1] = 0   ## remove
            links[rl1 * 16 + rd1][rl1 * 16 + rs1] = 0
            links[rl2 * 16 + rs2][rl2 * 16 + rd2] = 1   ## add
            links[rl2 * 16 + rd2][rl2 * 16 + rs2] = 1
            # check for islands
            edges_trial=[]
            temp=[]
            for i in range(0, 64):
                for j in range(0, 64):
                    temp.append(nodes[i])
                    temp.append(nodes[j])
                    if links[i][j] == 1:
                        temp.append(1)
                        # edges1.append(temp)
                    else:
                        temp.append(999)
                    edges_trial.append(temp)
                    # edges1.append(temp)
                    temp = []
            island=0
            for i in range(0, 64):
                p = str(dijkstra(edges_trial, nodes[rl1 * 16 + rs1], nodes[i]))
                q = str(dijkstra(edges_trial, nodes[rl1 * 16 + rd1], nodes[i]))
                p=p.split(',')
                q=q.split(',')
                cost_trial1=int(p[0][1:])
                cost_trial2=int(q[0][1:])
                if (cost_trial1>100 or cost_trial2>100):        ## cost is too high, resulting an isolated island, proving this change is not good, return changes back
                    island=1
                    break
            if (island==1): #reverse everything and restart
                links[rl1 * 16 + rs1][rl1 * 16 + rd1] = 1
                links[rl1 * 16 + rd1][rl1 * 16 + rs1] = 1
                links[rl2 * 16 + rs2][rl2 * 16 + rd2] = 0
                links[rl2 * 16 + rd2][rl2 * 16 + rs2] = 0
                continue
            # if reached till this point, then everything successful
            last_move[0]=1
            last_move[3]=rl1 * 16 + rs1     ## removed link source
            last_move[4]=rl1 * 16 + rd1     ## removed link destination
            last_move[5]=rl2 * 16 + rs2     ## added link source
            last_move[6]=rl2 * 16 + rd2     ## added link destination
            last_move[1]=-1
            last_move[2]=-1
            break
    
    return nodes, links

################################# reverse perturb ######################
## Turn everything back
def reverse_perturb(nodes,links,last_move):
    case=last_move[0]
    if case==0: #core swap
        rs=last_move[1]
        rd=last_move[2]
        t1 = nodes[rs]
        nodes[rs] = nodes[rd]
        nodes[rd] = t1  # re-exchanged nodes
    else:
        a=last_move[3]
        b=last_move[4]
        c=last_move[5]
        d=last_move[6]
        links[a][b] = 1
        links[b][a] = 1
        links[c][d] = 0
        links[d][c] = 0 #reversed links
    return nodes,links

################################# best perturb ##########################

def best_perturb(nodes,links, best_move):
    case = best_move[0]
    if case == 0:  # core swap
        rs = best_move[1]
        rd = best_move[2]
        t1 = nodes[rs]
        nodes[rs] = nodes[rd]
        nodes[rd] = t1
    else: #link place
        a = best_move[3]
        b = best_move[4]
        c = best_move[5]
        d = best_move[6]
        links[a][b] = 0
        links[b][a] = 0
        links[c][d] = 1
        links[d][c] = 1
    return nodes, links

################################### Reconfigure traffic file ####################################
def reconfigure_traffic(best_record, traffic_file, re_traffic_file):
    with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+ traffic_file+".stp") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        ## line 15 in stp file contains number of node, task, edge
        temp = content[15].split()
        num_node = int(temp[1])
        num_task = int(temp[2])
        num_edge = int(temp[3])
    ## write changed traffic into stp file
    t=''
    with open ("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+ re_traffic_file+".stp",'w') as q:
        for i in range(16):
            q.write(content[i])
            q.write('\n')
        for i in range(16, 16+num_task):    ## task info: change processor mapping
            temp1 = content[i].split()
            for j in range(0, num_node):
                if(int(temp1[1])==best_record[1][j]):
                    temp1[1] = str(j)
                    break
            
            t = ''
            for e in range(len(temp1)):
                t=t+str(temp1[e])+'  '
            q.write(t)
            q.write("\n")
            t = ''
        for i in range(16+num_task, 16+num_task+num_edge):    ## edge info: change src and dst processor mapping 
            temp2 = content[i].split()
            for j in range(0, num_node):
                if(int(temp2[3])==best_record[1][j]):
                    temp2[3] = str(j)
                    break
            for j in range(0, num_node):
                if(int(temp2[4])==best_record[1][j]):
                    temp2[4] = str(j)
                    break

            t = ''
            for e in range(len(temp2)):
                t=t+str(temp2[e])+'  '
            q.write(t)
            q.write("\n")
            t = ''

    with open ("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+re_traffic_file+"_mapping.txt", 'w') as r:
        r.write(str(best_record))

################################### Reconfigure architecture file ####################################
def reconfigure_architect(best_record, architecture_file, re_architecture_file):
    with open("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+architecture_file +".arch") as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        num_nodes = int(content[0])      ## line 0 is number of nodes
    ## write changed architecture into arch file
    t=''
    with open ("/content/drive/My Drive/Colaboratory/NoC_STAGE/"+re_architecture_file +".arch",'w') as q:
        q.write(content[0])
        q.write('\n')
        for i in range(1, 1+num_nodes):
            temp = content[i].split()
            if(best_record[1][i-1] == 0):
                temp[0] = str(0)
            elif(best_record[1][i-1] == 1):
                temp[0] = str(1)
            elif(best_record[1][i-1] == 2):
                temp[0] = str(6)
            elif(best_record[1][i-1] == 3):
                temp[0] = str(7)
            elif(best_record[1][i-1] == 4):
                temp[0] = str(11)
            elif(best_record[1][i-1] == 5):
                temp[0] = str(2)
            elif(best_record[1][i-1] == 6):
                temp[0] = str(4)
            elif(best_record[1][i-1] == 7):
                temp[0] = str(5)
            elif(best_record[1][i-1] == 8):
                temp[0] = str(8)
            elif(best_record[1][i-1] == 9):
                temp[0] = str(10)
            elif(best_record[1][i-1] == 10):
                temp[0] = str(3)
            elif(best_record[1][i-1] == 11):
                temp[0] = str(9)
            
            #temp[0] = str(best_record[1][i-1])
            t = ''
            for e in range(len(temp)):
                t=t+str(temp[e])+'  '
            q.write(t)
            q.write("\n")

################################ MAIN #####################################
def main(start_time):
    glo=0
    last_move = []
    best_move = []
    row = 2
    col = 4
    num_nodes = row * col
    for i in range(0, 7):
        last_move.append(-1)
        best_move.append(-1)
    num_links = 2 * row * col - row - col

    current_cost = 0            ######
    #current_sw_mean = 0
    #current_sw_dev = 0
    random.seed(1000)
    count1=0
    best_cost = 9999
    #best_mean = 9999
    #phv=-1
    #best_phv=0
    global_archive=[]
    local_archive=[] #complete information of NoC    local_archive contains 0:cost  1:nodes[]  2:links[][]  3:time elapsed
    local_pareto=[] #co-ordinates only
    train_set=[]
    labels=[]
    nodes=create_nodes(num_nodes) #create mesh
    links=create_mesh(row, col) #create mesh
    best_record = [9999, deepcopy(nodes)]            ## used to record the best solution

    app_name = "1"
    traffic_file = "Application_0" + app_name     ## here we assign traffic_file
    arch_traffic_file = "arch_Application_0" + app_name + "_cost"
    re_traffic_file = "reconfig_Application_0" + app_name + "_cost"
    architecture_file = "Heterogeneous_SoC_with_Ring_Topology"
    re_architecture_file = "reconfig_Application_0" + app_name + "_Heterogeneous_SoC_with_Ring_Topology_cost"

    arch = construct_architecture(architecture_file)
    time_to_stop = 1000     ## iters to stop the process

    make_traffic(arch, traffic_file, arch_traffic_file)
    traffic,injection=load_traffic() #load benchmark
    hop_count = calculate_hop_count(nodes, col)       ## create hop count matrix     ############
    cost = calculate_cost(traffic, hop_count, nodes)
    #mesh_mean, mesh_dev = calc_params(nodes,links,traffic)
    elapsed=timeit.default_timer()-start_time
    new_point = [0, 0, deepcopy(nodes), deepcopy(links),elapsed] #mesh
    new_pareto = [0] #shadows new_point
    local_pareto.append([0]) #shadows local_archive
    local_archive.append([0,deepcopy(nodes),deepcopy(links),elapsed]) #normalizing to remove absoluteness
    mesh_nodes=deepcopy(nodes)
    mesh_links=deepcopy(links)
    inp=[]
    for i in range(0,len(nodes)):
        inp.append(nodes[i])
    train_set.append(inp)
    labels.append(9999)
    num_try = 0
    f=0
    stop=0
    bias=0.01
    bx=0
    num_iter=10
    #w1=0.7
    #w2=0.3

    while 1:
        bx=bx+1
        count1=0
        for i in range(0,7):
            best_move[i]=-1
            last_move[i]=-1
        for a in range(0, num_iter):  # num_iter seperate perturbations from same starting point
            # make a perturbation
            nodes,links=perturb(nodes,links, 1, last_move)           ##randomly change nodes
            current_cost = calculate_cost(traffic, hop_count, nodes)
            #current_sw_mean, current_sw_dev = calc_params(nodes, links, traffic)

            #current_sw_mean=(1-current_sw_mean/mesh_mean)*w1        ## why (1-mean/mean)?
            #current_sw_dev=(1-current_sw_dev/mesh_dev)*w2
            new_pareto=[current_cost]
            #phv = phv_calculator(local_pareto, new_pareto)
            #if phv > best_phv:  # if current phv greater than current best phv i.e. successful perturbation
            if current_cost < best_cost:
                best_move=deepcopy(last_move)
                best_cost = current_cost
                #best_phv=phv
            else:
                ### optional: can add SA-like features here
                count1=count1+1         
            nodes,links=reverse_perturb(nodes,links, last_move)
        if(num_try == time_to_stop):
            print("best cost in the process is: ", best_record[0], "corresponding nodes is: ", best_record[1])      ## end of program, output the best result and reconfigure traffic file
            #reconfigure_traffic(best_record, traffic_file, re_traffic_file)
            reconfigure_architect(best_record, architecture_file, re_architecture_file)
            break

        if(count1==num_iter): #no improvement in num_iter perturbations, assume reached minima
            num_try += 1
            print("best_cost = ", best_cost, "num_try = ", num_try)
            print('********************************')
            temp_nodes, temp_links = best_perturb(nodes, links, best_move)
            if(best_cost < best_record[0]):
                best_record[0] = best_cost
                best_record[1] = temp_nodes
            bias = bias - 0.001
            bx=0
            ########### save the config #########
            elapsed=timeit.default_timer()-start_time
            asdq=0
            if(len(local_archive)>10):
                asdq=10
            else:
                asdq=len(local_archive)
            '''
            for b in range(0,asdq):
                text_file = open("drive/Colaboratory/NoC_STAGE/Output"+str(glo)+".txt", "w")
                glo=glo+1
                text_file.write("mean= %f \n" % local_archive[b][0])
                text_file.write("dev= %f \n" % local_archive[b][1])
                text_file.write("timestamp= %f \n" % local_archive[b][4])
                for i in range(0,16):
                    text_file.write("%s, " % local_archive[b][2][i])
                    #if i%16==15:
                    text_file.write("\n")
                text_file.write("\n")
                for i in range(0, 16):
                    for j in range(0, 16):
                        text_file.write("%d, " % local_archive[b][3][i][j])
                    text_file.write("\n")
                text_file.close()
            '''
            f=1
            global_archive.append(local_archive)
            local_archive=[]
            local_pareto=[]
            for i in range(0,len(labels)):
                if labels[i]==9999:
                    labels[i]=best_cost
            ######### send out for training ############

            regr = RandomForestRegressor(100)
            regr.fit(train_set, labels)

            stop=stop+1
            if(stop>=100):
                quit()

            ######### predict good start point ##########
            nodes = deepcopy(mesh_nodes)
            links = deepcopy(mesh_links)
            #mphv=best_phv
            #best_phv = 0
            mcost = best_cost
            best_cost = 9999
            c2=0
            while 1:
                n=random.randint(0,5)
                for i in range(0,n):
                    nodes,links=perturb(nodes,links,1,last_move)        ##100% change nodes
                tc = calculate_cost(traffic, hop_count, nodes)          ############
                #tm,td=calc_params(nodes,links,traffic)
                #tm=(1-tm/mesh_mean)*w1
                #td=(1-td/mesh_dev)*w2
                inpt=[]               ## because our label is cost, we can not use it as train set anymore
                for x in range(0,len(nodes)):           ## Core distribution statistics
                    #inpt.append(nodes[x])
                    if nodes[x]<1: #HQM
                       inpt.append(0)
                    elif nodes[x]<5: #CPU
                       inpt.append(1)
                    elif nodes[x]<10: #PE
                       inpt.append(2)
                    elif nodes[x]>9:  #DDR
                        inpt.append(3)

                predicted_end_cost=regr.predict([inpt])+bias #adding slight bias         ## Predict phv based on regression tree and new core map
                #print('predicted= ',predicted_end_mean, ' & mmean= ',mmean)
                if(predicted_end_cost<=mcost): # choose this as start point
                    local_pareto.append([tc])
                    elapsed=timeit.default_timer()-start_time
                    local_archive.append([tc, deepcopy(nodes),deepcopy(links),elapsed])     ########
                    break
                else:
                    c2=c2+1 #good starting point not found
                if c2==20:
                    break
            if c2==20: #if good starting points not found, pick random
                num_soln=0
                for i in range(0,len(global_archive)): # count total number of solutions obtained so far
                    num_soln = num_soln + len(global_archive[i])
                threshold=float(50)/num_soln
                r1 = random.random()
                if (r1<threshold): #pick random
                    nodes = deepcopy(mesh_nodes)
                    links = deepcopy(mesh_links)
                    for i in range(0,10): # make 10 random perturbation
                        nodes,links=perturb(nodes,links,1,last_move)
                    #tm,td=calc_params(nodes,links,traffic)
                    #tm=(1-tm/mesh_mean)*w1
                    #td=(1-td/mesh_dev)*w2
                    local_pareto.append([tc])
                    elapsed=timeit.default_timer()-start_time
                    local_archive.append([tc,deepcopy(nodes),deepcopy(links),elapsed])
                else: #pick from existing solutions
                    r2=random.randint(0,len(global_archive)-1)
                    r3=random.randint(0,len(global_archive[r2])-1)
                    local_archive.append(global_archive[r2][r3])
                    local_pareto.append([global_archive[r2][r3][0]])
        else: ## good solution found
            num_try += 1
            nodes, links = best_perturb(nodes, links,best_move)
            current_cost = calculate_cost(traffic, hop_count, nodes)
            #current_sw_mean, current_sw_dev = calc_params(nodes, links, traffic)
            print("a good solution's cost is: ", current_cost, "nodes are: ", nodes)
            #current_sw_mean = (1 - current_sw_mean / mesh_mean)*w1
            #current_sw_dev = (1 - current_sw_dev / mesh_dev)*w2
            ### new potential best candidate ready
            ## add candidate to local archive
            elapsed=timeit.default_timer()-start_time
            new_point = [current_cost, deepcopy(nodes), deepcopy(links), elapsed]
            new_pareto = [current_cost]
            i=0
            while len(local_pareto)>0:
                if (dominates(new_pareto, local_pareto[i], 0)):  # if new point completes dominates any existing point
                    del local_pareto[i]
                    del local_archive[i]
                    i=i-1
                i=i+1
                if (i>=len(local_archive)):
                    break
            local_pareto.append(new_pareto)
            local_archive.append(new_point)
            if (len(local_archive)>15): #optional: trimming to reduce size
                b_temp1 = []
                for i in range(0, len(local_archive)):
                    score=score=local_archive[i][0]  #0.7*(1-local_archive[i][0])+0.3*(1-local_archive[i][1]) #custom scoring function
                    b_temp1.append(score)
                b_num = numpy.array(b_temp1)
                b_ind = b_num.argsort()[:10]  # index of lowest n scores
                reduced_archive = []
                reduced_pareto=[]
                for i in range(0, len(b_ind)):
                    reduced_archive.append(local_archive[b_ind[i]])
                    reduced_pareto.append(local_pareto[b_ind[i]])
                local_archive = deepcopy(reduced_archive)
                local_pareto = deepcopy(reduced_pareto)

            #t2 = [current_sw_mean, current_sw_dev]
            t2 = []
            for x in range(0,len(nodes)):
                #inpt.append(nodes[x])
                if nodes[x]<1: #HQM
                    t2.append(0)
                elif nodes[x]<5: #CPU
                   t2.append(1)
                elif nodes[x]<10: #PE
                    t2.append(2)
                elif nodes[x]>9:  #DDR
                    t2.append(3)
            train_set.append(t2)            ##train_set contains mean, deviation and nodes distribution
            labels.append(9999)
            #labels.append(current_sw_mean)


start_time=timeit.default_timer()
main(start_time)
