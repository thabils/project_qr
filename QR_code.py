import math

import galois
import numpy as np
import matplotlib.pyplot as plt


class QR_code:
    def __init__(self, level, mask):
        self.level = level  # error correction level, can be 'L', 'M', 'Q', 'H'
        self.mask = mask  # the mask pattern, can be 'optimal' or either the three bits representing the mask in a list
        self.version = 6  # the version number, range from 1 to 40, only version number 6 is implemented

        # the generator polynomial of the Reed-Solomon code
        if level == 'L':
            self.generator = self.makeGenerator(2, 8, 9)
        elif level == 'M':
            self.generator = self.makeGenerator(2, 8, 8)
        elif level == 'Q':
            self.generator = self.makeGenerator(2, 8, 12)
        elif level == 'H':
            self.generator = self.makeGenerator(2, 8, 14)
        else:
            Exception('Invalid error correction level!')

        self.NUM_MASKS = 8  # the number of masks

    def encodeData(self, bitstream):
        # first add padding to the bitstream obtained from generate_dataStream()
        # then split this datasequence in blocks and perform RS-coding
        # and apply interleaving on the bytes (see the specification
        # section 8.6 'Constructing the final message codeword sequence')
        # INPUT:
        #  -bitstream: bitstream to be encoded in QR code. 1D numpy array e.g. bitstream=np.array([1,0,0,1,0,1,0,0,...])
        # OUTPUT:
        #  -data_enc: encoded bits after interleaving. Length should be 172*8 (version 6). 1D numpy array e.g. data_enc=np.array([1,0,0,1,0,1,0,0,...])
        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, 'bitstream must be a 1D numpy array'

        ################################################################################################################
        data_enc = np.array([])

        ################################################################################################################

        assert len(np.shape(data_enc)) == 1 and type(data_enc) is np.ndarray, 'data_enc must be a 1D numpy array'
        return data_enc

    def decodeData(self, data_enc):
        # Function to decode data, this is the inverse function of encodeData
        # INPUT:
        #  -data_enc: encoded binary data with the bytes being interleaved. 1D numpy array e.g. data_enc=np.array([1,0,0,1,0,1,0,0,...])
        #   length is equal to 172*8
        # OUTPUT:
        #  -bitstream: a bitstream with the padding removed. 1D numpy array e.g. bitstream=np.array([1,0,0,1,0,1,0,0,...])
        assert len(np.shape(data_enc)) == 1 and type(data_enc) is np.ndarray, 'data_enc must be a 1D numpy array'

        ################################################################################################################
        bitstream = np.array([])

        ################################################################################################################

        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, 'bitstream must be a 1D numpy array'
        return bitstream

    # QR-code generator/reader (do not change)
    def generate(self, data):
        # This function creates and displays a QR code matrix with either the optimal mask or a specific mask (depending on self.mask)
        # INPUT:
        #  -data: data to be encoded in the QR code. In this project a string with only characters from the alphanumeric mode
        #  e.g. data="A1 %"
        # OUTPUT:
        #  -QRmatrix: a 41 by 41 numpy array with 0's and 1's
        assert type(data) is str, 'data must be a string'

        bitstream = self.generate_dataStream(data)
        data_bits = self.encodeData(bitstream)

        if self.mask == 'optimal':
            # obtain optimal mask if mask=='optimal', otherwise use selected mask
            mask_code = [[int(x) for x in np.binary_repr(i, 3)] for i in range(self.NUM_MASKS)]
            score = np.ones(self.NUM_MASKS)
            score[:] = float('inf')
            for m in range(self.NUM_MASKS):
                QRmatrix_m = self.construct(data_bits, mask_code[m], show=False)
                score[m] = self.evaluateMask(QRmatrix_m)
                if score[m] == np.min(score):
                    QRmatrix = QRmatrix_m.copy()
                    self.mask = mask_code[m]

        # create the QR-code using either the selected or the optimal mask
        QRmatrix = self.construct(data_bits, self.mask)

        return QRmatrix

    def construct(self, data, mask, show=True):
        # This function creates a QR code matrix with specified data and
        # mask (this might not be the optimal mask though)
        # INPUT:
        #  -data: the output from encodeData, i.e. encoded bits after interleaving. Length should be 172*8 (version 6).
        #  1D numpy array e.g. data=np.array([1,0,0,1,0,1,0,0,...])
        #  -mask: three bits that represent the mask. 1D list e.g. mask=[1,0,0]
        # OUTPUT:
        #  -QRmatrix: a 41 by 41 numpy array with 0's and 1's
        L = 17 + 4 * self.version
        QRmatrix = np.zeros((L, L), dtype=int)

        PosPattern = np.ones((7, 7), dtype=int)
        PosPattern[[1, 5], 1:6] = 0
        PosPattern[1:6, [1, 5]] = 0

        QRmatrix[0:7, 0:7] = PosPattern
        QRmatrix[-7:, 0:7] = PosPattern
        QRmatrix[0:7, -7:] = PosPattern

        AlignPattern = np.ones((5, 5), dtype=int)
        AlignPattern[[1, 3], 1:4] = 0
        AlignPattern[1:4, [1, 3]] = 0

        QRmatrix[32:37, L - 9:L - 4] = AlignPattern

        L_timing = L - 2 * 8
        TimingPattern = np.zeros((1, L_timing), dtype=int)
        TimingPattern[0, 0::2] = np.ones((1, (L_timing + 1) // 2), dtype=int)

        QRmatrix[6, 8:(L_timing + 8)] = TimingPattern
        QRmatrix[8:(L_timing + 8), 6] = TimingPattern

        FI = self.encodeFormat(self.level, mask)
        FI = np.flip(FI)

        QRmatrix[0:6, 8] = FI[0:6]
        QRmatrix[7:9, 8] = FI[6:8]
        QRmatrix[8, 7] = FI[8]
        QRmatrix[8, 5::-1] = FI[9:]
        QRmatrix[8, L - 1:L - 9:-1] = FI[0:8]
        QRmatrix[L - 7:L, 8] = FI[8:]
        QRmatrix[L - 8, 8] = 1

        nogo = np.zeros((L, L), dtype=int)
        nogo[0:9, 0:9] = np.ones((9, 9), dtype=int)
        nogo[L - 1:L - 9:-1, 0:9] = np.ones((8, 9), dtype=int)
        nogo[0:9, L - 1:L - 9:-1] = np.ones((9, 8), dtype=int)
        nogo[6, 8:(L_timing + 8)] = np.ones((L_timing), dtype=int)
        nogo[8:(L_timing + 8), 6] = np.ones((1, L_timing), dtype=int)
        nogo[32:37, L - 9:L - 4] = np.ones((5, 5), dtype=int)
        nogo = np.delete(nogo, 6, 1)
        nogo = nogo[-1::-1, -1::-1];
        col1 = nogo[:, 0::2].copy()
        col2 = nogo[:, 1::2].copy()
        col1[:, 1::2] = col1[-1::-1, 1::2]
        col2[:, 1::2] = col2[-1::-1, 1::2]
        nogo_reshape = np.array([col1.flatten(order='F'), col2.flatten(order='F')])
        QR_reshape = np.zeros((2, np.shape(nogo_reshape)[1]), dtype=int)

        ind_col = 0
        ind_row = 0
        ind_data = 0

        for i in range(QR_reshape.size):
            if (nogo_reshape[ind_row, ind_col] == 0):
                QR_reshape[ind_row, ind_col] = data[ind_data]
                ind_data = ind_data + 1
                nogo_reshape[ind_row, ind_col] = 1

            ind_row = ind_row + 1
            if ind_row > 1:
                ind_row = 0
                ind_col = ind_col + 1

            if ind_data >= len(data):
                break

        QR_data = np.zeros((L - 1, L), dtype=int);
        colr = np.reshape(QR_reshape[0, :], (L, len(QR_reshape[0, :]) // L), order='F')
        colr[:, 1::2] = colr[-1::-1, 1::2]
        QR_data[0::2, :] = np.transpose(colr)

        coll = np.reshape(QR_reshape[1, :], (L, len(QR_reshape[1, :]) // L), order='F')
        coll[:, 1::2] = coll[-1::-1, 1::2]
        QR_data[1::2, :] = np.transpose(coll)

        QR_data = np.transpose(QR_data[-1::-1, -1::-1])
        QR_data = np.hstack((QR_data[:, 0:6], np.zeros((L, 1), dtype=int), QR_data[:, 6:]))

        QRmatrix = QRmatrix + QR_data

        QRmatrix[30:33, 0:2] = np.ones((3, 2), dtype=int)
        QRmatrix[29, 0] = 1

        nogo = nogo[-1::-1, -1::-1]
        nogo = np.hstack((nogo[:, 0:6], np.ones((L, 1), dtype=int), nogo[:, 6:]))

        QRmatrix = self.applyMask(mask, QRmatrix, nogo)

        if show == True:
            plt.matshow(QRmatrix, cmap='Greys')
            plt.show()

        return QRmatrix

    @staticmethod
    def read(QRmatrix):
        # function to read the encoded data from a QR code
        # INPUT:
        #  -QRmatrix: a 41 by 41 numpy array with 0's and 1's
        # OUTPUT:
        # -data_dec: data to be encoded in the QR code. In this project a string with only characters from the alphanumeric mode
        #  e.g. data="A1 %"
        assert np.shape(QRmatrix) == (41, 41) and type(QRmatrix) is np.ndarray, 'QRmatrix must be a 41 by numpy array'

        FI = np.zeros((15), dtype=int)
        FI[0:6] = QRmatrix[0:6, 8]
        FI[6:8] = QRmatrix[7:9, 8]
        FI[8] = QRmatrix[8, 7]
        FI[9:] = QRmatrix[8, 5::-1]
        FI = FI[-1::-1]

        L = np.shape(QRmatrix)[0]
        L_timing = L - 2 * 8

        [success, level, mask] = QR_code.decodeFormat(FI)

        if success:
            qr = QR_code(level, mask)
        else:
            FI = np.zeros((15), dtype=int)
            FI[0:8] = QRmatrix[8, L - 1:L - 9:-1]
            FI[8:] = QRmatrix[L - 7:L, 8]

            [success, level, mask] = QR_code.decodeFormat(FI)
            if success:
                qr = QR_code(level, mask)
            else:
                # print('Format information was not decoded succesfully')
                exit(-1)

        nogo = np.zeros((L, L))
        nogo[0:9, 0:9] = np.ones((9, 9), dtype=int)
        nogo[L - 1:L - 9:-1, 0:9] = np.ones((8, 9), dtype=int)
        nogo[0:9, L - 1:L - 9:-1] = np.ones((9, 8), dtype=int)

        nogo[6, 8:(L_timing + 8)] = np.ones((1, L_timing), dtype=int)
        nogo[8:(L_timing + 8), 6] = np.ones((L_timing), dtype=int)

        nogo[32:37, L - 9:L - 4] = np.ones((5, 5), dtype=int)

        QRmatrix = QR_code.applyMask(mask, QRmatrix, nogo)

        nogo = np.delete(nogo, 6, 1)
        nogo = nogo[-1::-1, -1::-1]
        col1 = nogo[:, 0::2]
        col2 = nogo[:, 1::2]
        col1[:, 1::2] = col1[-1::-1, 1::2]
        col2[:, 1::2] = col2[-1::-1, 1::2]

        nogo_reshape = np.vstack((np.transpose(col1.flatten(order='F')), np.transpose(col2.flatten(order='F'))))

        QRmatrix = np.delete(QRmatrix, 6, 1)
        QRmatrix = QRmatrix[-1::-1, -1::-1]
        col1 = QRmatrix[:, 0::2]
        col2 = QRmatrix[:, 1::2]
        col1[:, 1::2] = col1[-1::-1, 1::2]
        col2[:, 1::2] = col2[-1::-1, 1::2]

        QR_reshape = np.vstack((np.transpose(col1.flatten(order='F')), np.transpose(col2.flatten(order='F'))))

        data = np.zeros((172 * 8, 1))
        ind_col = 0
        ind_row = 0
        ind_data = 0
        for i in range(QR_reshape.size):
            if (nogo_reshape[ind_row, ind_col] == 0):
                data[ind_data] = QR_reshape[ind_row, ind_col]
                ind_data = ind_data + 1
                nogo_reshape[ind_row, ind_col] = 1

            ind_row = ind_row + 1
            if ind_row > 1:
                ind_row = 0
                ind_col = ind_col + 1

            if ind_data >= len(data):
                break

        bitstream = qr.decodeData(data.flatten())
        data_dec = QR_code.read_dataStream(bitstream)

        assert type(data_dec) is str, 'data_dec must be a string'
        return data_dec

    @staticmethod
    def generate_dataStream(data):
        # this function creates a bitstream from the user data.
        # ONLY IMPLEMENT ALPHANUMERIC MODE !!!!!!
        # INPUT:
        #  -data: the data string (for example 'ABC012')
        # OUTPUT:
        #  -bitstream: a 1D numpy array containing the bits that
        #  represent the input data, headers should be added, no padding must be added here.
        #  Add padding in EncodeData. e.g. bitstream=np.array([1,0,1,1,0,1,0,0,...])
        assert type(data) is str, 'data must be a string'

        ################################################################################################################
        # convert to ord() version of chars of string
        numbers = np.array([e for e in data]).view(np.int32)
        # convert to binary and combine
        bitstream = np.array([np.array([i for i in f'{e:08b}']) for e in numbers]).flatten()
        # split individual 1's and 0's

        ################################################################################################################

        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, 'bitstream must be a 1D numpy array'
        return bitstream

    @staticmethod
    def read_dataStream(bitstream):
        # inverse function of generate_dataStream: convert a bitstream to an alphanumeric string
        # INPUT:
        #  -bitstream: a 1D numpy array of bits (including the header bits) e.g. bitstream=np.array([1,0,1,1,0,1,0,0,...])
        # OUTPUT:
        #  -data: the encoded data string (for example 'ABC012')
        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, 'bitstream must be a 1D numpy array'

        ################################################################################################################
        # count amount (8 bit) chars
        n = int((len(bitstream) + 1) / 8)
        # make list of all seperate chars
        seperated_bits = (["".join(bitstream[index * 8:(index + 1) * 8]) for index in range(n)])
        # convert to int and then to char and join the answer
        data = "".join([chr(int(bits, 2)) for bits in seperated_bits])

        ################################################################################################################

        assert type(data) is str, 'data must be a string'
        return data

    @staticmethod
    def encodeFormat(level, mask):
        # Encodes the 5 bit format to a 15 bit sequence using a BCH code
        # INPUT:
        #  -level: specified level 'L','M','Q' or 'H'
        #  -mask: three bits that represent the mask. 1D list e.g. mask=[1,0,0]
        # OUTPUT:
        # format: 1D numpy array with the FI-codeword, with the special FI-mask applied (see specification)
        assert type(mask) is list and len(mask) == 3, 'mask must be a list of length 3'

        ################################################################################################################
        format = np.array([])
        ################################################################################################################

        assert len(np.shape(format)) == 1 and type(
            format) is np.ndarray and format.size == 15, 'format must be a 1D numpy array of length 15'
        return format

    @staticmethod
    def decodeFormat(Format):
        # Decode the format information
        # INPUT:
        # -format: 1D numpy array (15bits) with format information (with FI-mask applied)
        # OUTPUT:
        # -success: True if decodation succeeded, False if decodation failed
        # -level: being an element of {'L','M','Q','H'}
        # -mask: three bits that represent the mask. 1D list e.g. mask=[1,0,0]
        assert len(np.shape(Format)) == 1 and type(
            Format) is np.ndarray and Format.size == 15, 'format must be a 1D numpy array of length 15'

        ################################################################################################################
        mask = [-1, -1, -1]
        success = False
        level = ""
        ################################################################################################################

        assert type(mask) is list and len(mask) == 3, 'mask must be a list of length 3'
        return success, level, mask

    @staticmethod
    def makeGenerator(p, m, t):
        # Generate the Reed-Solomon generator polynomial with error correcting capability t over GF(p^m)
        # INPUT:
        #  -p: field characteristic, prime number
        #  -m: positive integer
        #  -t: error correction capability of the Reed-Solomon code, positive integer > 1
        # OUTPUT:
        #  -generator: galois.Poly object representing the generator polynomial

        ################################################################################################################
        generator = None

        ################################################################################################################

        assert type(generator) == type(galois.Poly([0], field=galois.GF(m))), 'generator must be a galois.Poly object'
        return generator

    @staticmethod
    def encodeRS(informationword, p, m, n, k, generator):
        # Encode the informationword
        # INPUT:
        #  -informationword: a 1D array of galois.GF elements that represents the information word coefficients in GF(p^m) (first element is the highest degree coefficient)
        #  -p: field characteristic, prime number
        #  -m: positive integer
        #  -n: codeword length, <= p^m-1
        #  -k: information word length
        #  -generator: galois.Poly object representing the generator polynomial
        # OUTPUT:
        #  -codeword: a 1D array of galois.GF elements that represents the codeword coefficients in GF(p^m) corresponding to systematic Reed-Solomon coding of the corresponding information word (first element is the highest degree coefficient)
        prim_poly = galois.primitive_poly(p, m)
        GF = galois.GF(p ** m, irreducible_poly=prim_poly)
        assert type(informationword) is GF and len(
            np.shape(informationword)) == 1, 'each element of informationword(1D)  must be a galois.GF element'
        assert type(generator) == type(galois.Poly([0], field=galois.GF(m))), 'generator must be a galois.Poly object'

        ################################################################################################################
        codeword = []

        ################################################################################################################

        assert type(codeword) is GF and len(
            np.shape(codeword)) == 1, 'each element of codeword(1D)  must be a galois.GF element'
        return codeword

    @staticmethod
    def decodeRS(codeword, p, m, n, k, generator):
        # Decode the codeword
        # INPUT:
        #  -codeword: a 1D array of galois.GF elements that represents the codeword coefficients in GF(p^m) corresponding to systematic Reed-Solomon coding of the corresponding information word (first element is the highest degree coefficient)
        #  -p: field characteristic, prime number
        #  -m: positive integer
        #  -n: codeword length, <= p^m-1
        #  -k: decoded word length
        #  -generator: galois.Poly object representing the generator polynomial
        # OUTPUT:
        #  -decoded: a 1D array of galois.GF elements that represents the decoded information word coefficients in GF(p^m) (first element is the highest degree coefficient)
        prim_poly = galois.primitive_poly(p, m)
        GF = galois.GF(p ** m, irreducible_poly=prim_poly)
        assert type(codeword) is GF and len(
            np.shape(codeword)) == 1, 'each element of codeword(1D)  must be a galois.GF element'
        assert type(generator) == type(galois.Poly([0], field=galois.GF(m))), 'generator must be a galois.Poly object'

        ################################################################################################################
        decoded = []

        ################################################################################################################

        assert type(decoded) is GF and len(
            np.shape(decoded)) == 1, 'each element of decoded(1D)  must be a galois.GF element'
        return decoded

    # function to mask or unmask a QR_code matrix and to evaluate the masked QR symbol (do not change)
    @staticmethod
    def applyMask(mask, QRmatrix, nogo):
        # define all the masking functions
        maskfun1 = lambda i, j: (i + j) % 2 == 0
        maskfun2 = lambda i, j: (i) % 2 == 0
        maskfun3 = lambda i, j: (j) % 3 == 0
        maskfun4 = lambda i, j: (i + j) % 3 == 0
        maskfun5 = lambda i, j: (math.floor(i / 2) + math.floor(j / 3)) % 2 == 0
        maskfun6 = lambda i, j: (i * j) % 2 + (i * j) % 3 == 0
        maskfun7 = lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0
        maskfun8 = lambda i, j: ((i + j) % 2 + (i * j) % 3) % 2 == 0

        maskfun = [maskfun1, maskfun2, maskfun3, maskfun4, maskfun5, maskfun6, maskfun7, maskfun8]

        L = len(QRmatrix)
        QRmatrix_masked = QRmatrix.copy()

        mask_number = int(''.join(str(el) for el in mask), 2)

        maskfunction = maskfun[mask_number]

        for i in range(L):
            for j in range(L):
                if nogo[i, j] == 0:
                    QRmatrix_masked[i, j] = (QRmatrix[i, j] + maskfunction(i, j)) % 2

        return QRmatrix_masked

    @staticmethod
    def evaluateMask(QRmatrix):
        Ni = [3, 3, 40, 10]
        L = len(QRmatrix)
        score = 0
        QRmatrix_temp = np.vstack((QRmatrix, 2 * np.ones((1, L)), np.transpose(QRmatrix), 2 * np.ones((1, L))))

        vector = QRmatrix_temp.flatten(order='F')
        splt = QR_code.SplitVec(vector)

        neighbours = np.array([len(x) for x in splt])
        temp = neighbours > 5
        if (temp).any():
            score += sum([x - 5 + Ni[0] for x in neighbours if x > 5])

        QRmatrix_tmp = QRmatrix
        rec_sizes = np.array([[5, 2, 4, 4, 3, 4, 2, 3, 2, 3, 2], [2, 5, 4, 3, 4, 2, 4, 3, 3, 2, 2]])

        for i in range(np.shape(rec_sizes)[1]):
            QRmatrix_tmp, num = QR_code.find_rect(QRmatrix_tmp, rec_sizes[0, i], rec_sizes[1, i])
            score += num * (rec_sizes[0, i] - 1) * (rec_sizes[1, i] - 1) * Ni[1]

        QRmatrix_tmp = np.vstack((QRmatrix, 2 * np.ones((1, L)), np.transpose(QRmatrix), 2 * np.ones((1, L))))
        temp = QRmatrix_tmp.flatten(order='F')
        temp2 = [x for x in range(len(temp) - 6) if (temp[x:x + 7] == [1, 0, 1, 1, 1, 0, 1]).all()]
        score += Ni[2] * len(temp2)

        nDark = sum(sum(QRmatrix == 1)) / L ** 2
        k = math.floor(abs(nDark - 0.5) / 0.05)
        score += Ni[3] * k

        return score

    @staticmethod
    def SplitVec(vector):
        output = []
        temp = np.where(np.diff(vector) != 0)[0]
        temp = temp + 1
        temp = np.insert(temp, 0, 0)

        for i in range(len(temp)):
            if i == len(temp) - 1:
                output.append(vector[temp[i]:])
            else:
                output.append(vector[temp[i]:temp[i + 1]])

        return output

    @staticmethod
    def find_rect(A, nR, nC):

        Lx = np.shape(A)[0]
        Ly = np.shape(A)[1]
        num = 0
        A_new = A.copy()

        for x in range(Lx - nR + 1):
            for y in range(Ly - nC + 1):
                test = np.unique(A_new[x:x + nR, y:y + nC])

                if len(test) == 1:
                    num += 1
                    A_new[x:x + nR, y:y + nC] = np.reshape(np.arange(2 + x * nR + y, 2 + nR * nC + x * nR + y),
                                                           (nR, nC))

        return A_new, num
