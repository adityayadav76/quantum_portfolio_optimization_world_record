import requests
import datetime 

class AutomatskiInitiumTabuSolver:
    
    def __init__(self, host, port, max_iter=1000, tabu_tenure=10, timeout=3600, apiKey:str = "open"): #timeout in seconds
        self.apiKey = apiKey 
        self.host = host
        self.port = port
        self.max_iter= max_iter
        self.tabu_tenure= tabu_tenure 
        self.timeout = timeout 

    def solve(self, quboDict, silent=False):
        self.silent = silent
        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0
        
        '''
        try: 
            if isinstance(quboDict[0], dict):
                quboDict = quboDict[0]
        except:
            pass         
        '''
        
        qubo = []
        for key in quboDict:
            m = self.index(key[0])
            n = self.index(key[1])
            i = min(m,n) # i always has to be lower or equal to j. 
            j = max(m,n)
            v = float(quboDict[key])
            qubo.append( [i,j,v] )
        
        #print(qubo)
        if not self.silent: print("Executing Annealer With ...")
        if not self.silent: print(f"{len(self.indexToKeys.keys())} Qubits and...")
        if not self.silent: print(f"{len(qubo)} clauses")
        
        tstart = datetime.datetime.now()
        
        r = requests.post(f'http://{self.host}:{self.port}/api/tabu', json={'max_iter': self.max_iter, 'tabu_tenure': self.tabu_tenure, 'timeout': self.timeout, 'qubo': qubo, 'apiKey':self.apiKey}, timeout=None)
        #print(f"Status Code: {r.status_code}, Response: {r.json()}")
        #print(self.keysToIndex)
        #print(self.indexToKeys)
        
        tend = datetime.datetime.now()
        if not self.silent: print(f"Time Taken {(tend - tstart)}")
        
        struct = r.json()
        cannotProceed = None
        try:
            if struct["error"] :
                print(struct["error"])
                cannotProceed = True
                raise Exception(struct["error"])
        except Exception as e:
            if cannotProceed:
                raise e            
            pass            
        
        
        bits = struct['bits']
        value = struct['value']
        answer = {}
        for bit in bits:
            index = int(bit)
            answer[self.indexToKeys[index]] = bits[bit]
        
        #print(f'count:{self.count}')
        
        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0
        
        return answer, value
        
    def index(self, key):
        if key in self.keysToIndex:
            return self.keysToIndex[key]
        else:
            self.keysToIndex[key] = self.count
            self.indexToKeys[self.count] = key
            self.count = self.count + 1
            return self.count - 1
            
            
class AutomatskiInitiumSASolver:
    
    def __init__(self, host, port, max_iter=1000, temp=10.0, cooling_rate=0.01, num_reads=10, timeout=3600, apiKey:str = "open"): # timeout in seconds 
        self.apiKey = apiKey
        self.host = host
        self.port = port
        self.max_iter= max_iter
        self.temp = temp
        self.cooling_rate = cooling_rate
        self.num_reads = num_reads
        self.timeout = timeout
        
    def solve(self, quboDict, silent=False):
        self.silent = silent
        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0
        
        '''
        try: 
            if isinstance(quboDict[0], dict):
                quboDict = quboDict[0]
        except:
            pass            
        '''
        
        qubo = []
        for key in quboDict:
            m = self.index(key[0])
            n = self.index(key[1])
            i = min(m,n) # i always has to be lower or equal to j. 
            j = max(m,n)
            v = float(quboDict[key])
            qubo.append( [i,j,v] )
        
        #print(qubo)
        
        if not self.silent: print("Executing Annealer With ...")
        if not self.silent: print(f"{len(self.indexToKeys.keys())} Qubits")
        if not self.silent: print(f"{len(qubo)} clauses")
        
        tstart = datetime.datetime.now()
        
        r = requests.post(f'http://{self.host}:{self.port}/api/sa', json={'max_iter': self.max_iter, 'temp': self.temp, 'num_reads': self.num_reads,'cooling_rate': self.cooling_rate,'timeout': self.timeout, 'qubo': qubo, 'apiKey': self.apiKey})
        #print(f"Status Code: {r.status_code}, Response: {r.json()}")
        #print(self.keysToIndex)
        #print(self.indexToKeys)
        
        tend = datetime.datetime.now()
        if not self.silent: print(f"Time Taken {(tend - tstart)}")
        
        struct = r.json()
        cannotProceed = None
        try:
            if struct["error"] :
                print(struct["error"])
                cannotProceed = True
                raise Exception(struct["error"])
        except Exception as e:
            if cannotProceed:
                raise e            
            pass
            
        bits = struct['bits']
        value = struct['value']
        answer = {}
        for bit in bits:
            index = int(bit)
            answer[self.indexToKeys[index]] = bits[bit]
        
        #print(f'count:{self.count}')
        
        self.keysToIndex = {}
        self.indexToKeys = {}
        self.count = 0
        
        return answer, value
        
    def index(self, key):
        if key in self.keysToIndex:
            return self.keysToIndex[key]
        else:
            self.keysToIndex[key] = self.count
            self.indexToKeys[self.count] = key
            self.count = self.count + 1
            return self.count - 1  

