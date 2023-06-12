# this escode works in GF(2^8), i.e. work in bytes
import numpy as np
import random

# xor for add in GF(2^8)
def GF_add(a, b):
    return a ^ b

# xor for minus in GF(2^8)
def GF_minus(a, b):
    return a ^ b

# a * b = 2 ^ (log a + log b) in GF(2^8)
def GF_multiply(a, b):
    if a == 0 or b == 0:
        return 0

    return power[(log[a] + log[b]) % 255]

# a / b = 2 ^ (log a - log b) in GF(2^8)
def GF_devide(a, b):
    return power[(log[a] - log[b] + 255) % 255]

# a ** b in GF(2^8)
def GF_power(a, b):
    if b == 0:
        return 1
    
    if b == 1:
        return a
    
    res = a
    times = 1
    while times < b:
        res = GF_multiply(res, a)
        times += 1

    return res

# generate power and log tables for GF(2^8)
power = np.zeros(256, dtype=int)
log = np.zeros(256, dtype=int)
n = 1
for i in range(0, 256):
    power[i] = n
    log[n] = i

    n *= 2

    # modular by the prime polynomial: P_8(x) = x^8 + x^4 + x^3 + x^2 + 1
    if n >= 256:
        n = n ^ 0x11d

log[1] = 0 # log[1] is 255, but it should be 0

# generate multiply table for GF(2^8)
multiply = np.zeros((256, 256), dtype=int)
for i in range(256):
    for j in range(256):
        multiply[i][j] = GF_multiply(i, j)

# generate inverse table for GF(2^8)
inverse = np.zeros(256, dtype=int)
for i in range(256):
    for j in range(256):
        if multiply[i][j] == 1:
            inverse[i] = j

# generate encode matrix (k+m)*k using vandermonde matrix
def gen_encode_matrix_vandermonde(k, m):
    # generate I
    I = np.zeros((k, k), dtype=int)
    for i in range(k):
        I[i][i] = 1

    # generate rows of vandermonde matrix
    vandermonde = np.zeros((m, k), dtype=int)
    for i in range(m):
        for j in range(k):
            vandermonde[i][j] = GF_power(i+1, j)
    
    encode_matrix = np.concatenate([I, vandermonde])

    return encode_matrix

# generate encode matrix (k+m)*k using cauchy matrix
def gen_encode_matrix_cauchy(k, m):
    # generate I
    I = np.zeros((k, k), dtype=int)
    for i in range(k):
        I[i][i] = 1

    # generate rows of cauchy matrix
    cauchy = np.zeros((m, k), dtype=int)
    for i in range(m):
        for j in range(k):
            cauchy[i][j] = inverse[GF_add(i, j)]

    encode_matrix = np.concatenate([I, cauchy])

    return encode_matrix

# generate decode matrix k*k according to missing rows
def gen_decode_matrix_vandermonde(encode_matrix, missing_rows):
    # drop rows in encode matrix
    reconstruct_encode_matrix = np.delete(encode_matrix, missing_rows, axis=0)
    reconstruct_encode_matrix = reconstruct_encode_matrix[:encode_matrix.shape[1], :]

    print("reconstruct_encode_matrix (vandermonde version):\n", reconstruct_encode_matrix)

    # get decode matrix
    flag, reconstruct_decode_matrix = get_inv_matrix(reconstruct_encode_matrix)

    return flag, reconstruct_decode_matrix

def gen_decode_matrix_cauchy(encode_matrix, missing_rows):
    # drop rows in encode matrix
    reconstruct_encode_matrix = np.delete(encode_matrix, missing_rows, axis=0)
    reconstruct_encode_matrix = reconstruct_encode_matrix[:encode_matrix.shape[1], :]

    print("reconstruct_encode_matrix (cauchy version):\n", reconstruct_encode_matrix)

    # get decode matrix
    flag, reconstruct_decode_matrix = get_inv_matrix(reconstruct_encode_matrix)

    return flag, reconstruct_decode_matrix

# calculate inv matrix
def get_inv_matrix(matrix):
    # calculate inv matrix using gaussian elimination
    A = matrix
    n = matrix.shape[0]
    
    # inv initialize to I
    inv = np.zeros((n, n), dtype=int)
    for i in range(n):
        inv[i][i] = 1

    for i in range(n):
        # find a row with main element = 0 (element on main diag)
        # swap that row with another row with non-zero element in the same column
        if A[i][i] == 0:
            j = i + 1
            while j < n:
                if A[j][i] != 0:
                    break
                j += 1
            
            if j == n:
                # it's a singular matrix
                return False, []
            
            # swap both A and inv
            A[[i, j], :] = A[[j, i], :]
            inv[[i, j], :] = inv[[j, i], :]

        # scale main element to 1
        # main element = A[i][i] = e
        if A[i][i] != 1:
            # get 1 / e
            e = inverse[A[i][i]]
            # scale current row (multiply 1 / e)
            for j in range(n):
                A[i][j] = GF_multiply(A[i][j], e)
                inv[i][j] = GF_multiply(inv[i][j], e)

        # make all elements (except the main element) in that column become 0 
        for j in range(n):
            if j == i:
                continue

            # v = scale rate
            v = A[j][i]
            if v != 0:
                for k in range(n):
                    A[j][k] = GF_add(A[j][k], GF_multiply(v, A[i][k]))
                    inv[j][k] = GF_add(inv[j][k], GF_multiply(v, inv[i][k]))

    return True, inv

# matrix multiply in GF(2^8)
# A is matrix, B is vector
def matrix_multiply(A, B):
    if A.shape[1] != B.shape[0]:
        return None

    vec = []
    for i in range(A.shape[0]):
        res = 0
        for j in range(A.shape[1]):
            res = GF_add(res, GF_multiply(A[i][j], B[j]))
        vec.append(res)

    return np.array(vec).transpose()

# drop parts of encode data
# drops should be a list of row indices
def encode_data_drop(encode_data, m, drops):
    if len(drops) > m:
        return None
    
    incomplete_data = np.delete(encode_data, drops)

    return incomplete_data

# reedsolomon API
def reedsolomon(k, m, raw_data, drops, matrix_type):
    if matrix_type not in ["vandermonde", "cauchy"]:
        print("matrix type error")
        return False, []
    
    if k <= 0 or m <= 0:
        print("k and m should not be 0")
        return False, []
    
    if raw_data.shape[0] != k:
        print("raw_data size != k")
        return False, []
    
    if drops.shape[0] > m:
        print("dropping too much sets of encoded data")
        return False, []
    
    if drops.shape[0] != len(set(drops)):
        print("dropping the duplicated sets of data")
        return False, []
    
    if matrix_type == "vandermonde":
        # using vandermonde version

        # generate encode matrix using vandermonde matrix
        em = gen_encode_matrix_vandermonde(k, m)
        print("encode matrix (vandermonde version):\n", em)

        # encode the raw data
        encode_data = matrix_multiply(em, raw_data)
        print("encoded data (vandermonde version):", encode_data)

        # drop specified sets of the encoded data
        incomplete_data = encode_data_drop(encode_data, m, drops)
        print("data after lost (vandermonde version):", incomplete_data)

        # generate decode matrix and reconstruct raw data (vandermonde)
        flag, dm = gen_decode_matrix_vandermonde(em, drops)

        if flag == False:
            print("decode matrix (vandermonde version) is not invertible!")
        else:
            print("decode matrix (vandermonde version):\n", dm)
            reconstruct_data = matrix_multiply(dm, incomplete_data)
            print("reconstructed data (vandermonde version):", reconstruct_data)

    else:
        # using cauchy version

        # generate encode matrix using cauchy matrix
        em = gen_encode_matrix_cauchy(k, m)
        print("encode matrix (cauchy version):\n", em)

        # encode the raw data
        encode_data = matrix_multiply(em, raw_data)
        print("encoded data (cauchy version):", encode_data)

        # drop specified sets of the encoded data
        incomplete_data = encode_data_drop(encode_data, m, drops)
        print("data after lost (cauchy version):", incomplete_data)

        # generate decode matrix and reconstruct raw data (cauchy)
        flag, dm = gen_decode_matrix_vandermonde(em, drops)

        if flag == False:
            print("decode matrix (cauchy version) is not invertible!")
        else:
            print("decode matrix (cauchy version):\n", dm)
            reconstruct_data = matrix_multiply(dm, incomplete_data)
            print("reconstructed data (cauchy version):", reconstruct_data)

    return flag, dm


if __name__ == '__main__':
    # randomly choose k and m
    k = np.random.randint(3, 11)
    m = np.random.randint(1, k+1)
    print("k:", k, "m:", m)

    # randomly generate raw data
    raw_data = np.array(np.random.randint(0, 256, k)).transpose()
    print("raw_data:", raw_data)

    # randomly drop m sets of encoded data
    drops = np.array(random.sample(range(0, k+m), m))
    print("drops:", drops)

    # vandermonde
    flag_v, re_data_v = reedsolomon(k, m, raw_data, drops, "vandermonde")
    # cauchy
    flag_c, re_data_c = reedsolomon(k, m, raw_data, drops, "cauchy")