\ Model Pediatrician Scheduling
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  CW[1,1] + CW[1,2] + CW[1,3] + CW[1,4] + CW[1,5] + CW[1,6] + CW[1,7]
   + CW[1,8] + CW[1,9] + CW[1,10] + CW[1,11] + CW[1,12] + CW[1,13]
   + CW[2,1] + CW[2,2] + CW[2,3] + CW[2,4] + CW[2,5] + CW[2,6] + CW[2,7]
   + CW[2,8] + CW[2,9] + CW[2,10] + CW[2,11] + CW[2,12] + CW[2,13]
   + E2[1,2,6] + E2[1,4,8] + E2[1,6,10] + E2[1,8,12] + E2[1,10,14]
   + E2[1,16,20] + E2[1,18,22] + E2[1,20,24] + E2[1,22,26] + E2[1,24,28]
   + E2[1,30,34] + E2[1,32,36] + E2[1,34,38] + E2[1,36,40] + E2[1,38,42]
   + E2[1,12,16] + E2[1,26,30] + E2[2,2,6] + E2[2,4,8] + E2[2,6,10]
   + E2[2,8,12] + E2[2,10,14] + E2[2,16,20] + E2[2,18,22] + E2[2,20,24]
   + E2[2,22,26] + E2[2,24,28] + E2[2,30,34] + E2[2,32,36] + E2[2,34,38]
   + E2[2,36,40] + E2[2,38,42] + E2[2,12,16] + E2[2,26,30] + E2[3,2,6]
   + E2[3,4,8] + E2[3,6,10] + E2[3,8,12] + E2[3,10,14] + E2[3,16,20]
   + E2[3,18,22] + E2[3,20,24] + E2[3,22,26] + E2[3,24,28] + E2[3,30,34]
   + E2[3,32,36] + E2[3,34,38] + E2[3,36,40] + E2[3,38,42] + E2[3,12,16]
   + E2[3,26,30] + E2[4,2,6] + E2[4,4,8] + E2[4,6,10] + E2[4,8,12]
   + E2[4,10,14] + E2[4,16,20] + E2[4,18,22] + E2[4,20,24] + E2[4,22,26]
   + E2[4,24,28] + E2[4,30,34] + E2[4,32,36] + E2[4,34,38] + E2[4,36,40]
   + E2[4,38,42] + E2[4,12,16] + E2[4,26,30] + E2[5,2,6] + E2[5,4,8]
   + E2[5,6,10] + E2[5,8,12] + E2[5,10,14] + E2[5,16,20] + E2[5,18,22]
   + E2[5,20,24] + E2[5,22,26] + E2[5,24,28] + E2[5,30,34] + E2[5,32,36]
   + E2[5,34,38] + E2[5,36,40] + E2[5,38,42] + E2[5,12,16] + E2[5,26,30]
   + E2[6,2,6] + E2[6,4,8] + E2[6,6,10] + E2[6,8,12] + E2[6,10,14]
   + E2[6,16,20] + E2[6,18,22] + E2[6,20,24] + E2[6,22,26] + E2[6,24,28]
   + E2[6,30,34] + E2[6,32,36] + E2[6,34,38] + E2[6,36,40] + E2[6,38,42]
   + E2[6,12,16] + E2[6,26,30] + E2[7,2,6] + E2[7,4,8] + E2[7,6,10]
   + E2[7,8,12] + E2[7,10,14] + E2[7,16,20] + E2[7,18,22] + E2[7,20,24]
   + E2[7,22,26] + E2[7,24,28] + E2[7,30,34] + E2[7,32,36] + E2[7,34,38]
   + E2[7,36,40] + E2[7,38,42] + E2[7,12,16] + E2[7,26,30] + E2[8,2,6]
   + E2[8,4,8] + E2[8,6,10] + E2[8,8,12] + E2[8,10,14] + E2[8,16,20]
   + E2[8,18,22] + E2[8,20,24] + E2[8,22,26] + E2[8,24,28] + E2[8,30,34]
   + E2[8,32,36] + E2[8,34,38] + E2[8,36,40] + E2[8,38,42] + E2[8,12,16]
   + E2[8,26,30] + E2[9,2,6] + E2[9,4,8] + E2[9,6,10] + E2[9,8,12]
   + E2[9,10,14] + E2[9,16,20] + E2[9,18,22] + E2[9,20,24] + E2[9,22,26]
   + E2[9,24,28] + E2[9,30,34] + E2[9,32,36] + E2[9,34,38] + E2[9,36,40]
   + E2[9,38,42] + E2[9,12,16] + E2[9,26,30] + E2[10,2,6] + E2[10,4,8]
   + E2[10,6,10] + E2[10,8,12] + E2[10,10,14] + E2[10,16,20] + E2[10,18,22]
   + E2[10,20,24] + E2[10,22,26] + E2[10,24,28] + E2[10,30,34]
   + E2[10,32,36] + E2[10,34,38] + E2[10,36,40] + E2[10,38,42]
   + E2[10,12,16] + E2[10,26,30] + E2[11,2,6] + E2[11,4,8] + E2[11,6,10]
   + E2[11,8,12] + E2[11,10,14] + E2[11,16,20] + E2[11,18,22]
   + E2[11,20,24] + E2[11,22,26] + E2[11,24,28] + E2[11,30,34]
   + E2[11,32,36] + E2[11,34,38] + E2[11,36,40] + E2[11,38,42]
   + E2[11,12,16] + E2[11,26,30] + E2[12,2,6] + E2[12,4,8] + E2[12,6,10]
   + E2[12,8,12] + E2[12,10,14] + E2[12,16,20] + E2[12,18,22]
   + E2[12,20,24] + E2[12,22,26] + E2[12,24,28] + E2[12,30,34]
   + E2[12,32,36] + E2[12,34,38] + E2[12,36,40] + E2[12,38,42]
   + E2[12,12,16] + E2[12,26,30] + E2[13,2,6] + E2[13,4,8] + E2[13,6,10]
   + E2[13,8,12] + E2[13,10,14] + E2[13,16,20] + E2[13,18,22]
   + E2[13,20,24] + E2[13,22,26] + E2[13,24,28] + E2[13,30,34]
   + E2[13,32,36] + E2[13,34,38] + E2[13,36,40] + E2[13,38,42]
   + E2[13,12,16] + E2[13,26,30] + E3[1,1] + E3[1,2] + E3[1,3] + E3[1,4]
   + E3[1,5] + E3[1,6] + E3[1,7] + E3[1,8] + E3[1,9] + E3[1,10] + E3[1,11]
   + E3[1,12] + E3[1,13] + E3[2,1] + E3[2,2] + E3[2,3] + E3[2,4] + E3[2,5]
   + E3[2,6] + E3[2,7] + E3[2,8] + E3[2,9] + E3[2,10] + E3[2,11] + E3[2,12]
   + E3[2,13] + E3[3,1] + E3[3,2] + E3[3,3] + E3[3,4] + E3[3,5] + E3[3,6]
   + E3[3,7] + E3[3,8] + E3[3,9] + E3[3,10] + E3[3,11] + E3[3,12]
   + E3[3,13] + Dplus[1] + Dplus[2] + Dplus[3] + Dplus[4] + Dplus[5]
   + Dplus[6] + Dplus[7] + Dplus[8] + Dplus[9] + Dplus[10] + Dplus[11]
   + Dplus[12] + Dplus[13] + Dminus[1] + Dminus[2] + Dminus[3] + Dminus[4]
   + Dminus[5] + Dminus[6] + Dminus[7] + Dminus[8] + Dminus[9] + Dminus[10]
   + Dminus[11] + Dminus[12] + Dminus[13]
Subject To
 onePed[1]: X[1,1] + X[1,2] + X[1,3] + X[1,4] + X[1,5] + X[1,6] + X[1,7]
   + X[1,8] + X[1,9] + X[1,10] + X[1,11] + X[1,12] + X[1,13] = 1
 onePed[2]: X[2,1] + X[2,2] + X[2,3] + X[2,4] + X[2,5] + X[2,6] + X[2,7]
   + X[2,8] + X[2,9] + X[2,10] + X[2,11] + X[2,12] + X[2,13] = 1
 onePed[3]: X[3,1] + X[3,2] + X[3,3] + X[3,4] + X[3,5] + X[3,6] + X[3,7]
   + X[3,8] + X[3,9] + X[3,10] + X[3,11] + X[3,12] + X[3,13] = 1
 onePed[4]: X[4,1] + X[4,2] + X[4,3] + X[4,4] + X[4,5] + X[4,6] + X[4,7]
   + X[4,8] + X[4,9] + X[4,10] + X[4,11] + X[4,12] + X[4,13] = 1
 onePed[5]: X[5,1] + X[5,2] + X[5,3] + X[5,4] + X[5,5] + X[5,6] + X[5,7]
   + X[5,8] + X[5,9] + X[5,10] + X[5,11] + X[5,12] + X[5,13] = 1
 onePed[6]: X[6,1] + X[6,2] + X[6,3] + X[6,4] + X[6,5] + X[6,6] + X[6,7]
   + X[6,8] + X[6,9] + X[6,10] + X[6,11] + X[6,12] + X[6,13] = 1
 onePed[7]: X[7,1] + X[7,2] + X[7,3] + X[7,4] + X[7,5] + X[7,6] + X[7,7]
   + X[7,8] + X[7,9] + X[7,10] + X[7,11] + X[7,12] + X[7,13] = 1
 onePed[8]: X[8,1] + X[8,2] + X[8,3] + X[8,4] + X[8,5] + X[8,6] + X[8,7]
   + X[8,8] + X[8,9] + X[8,10] + X[8,11] + X[8,12] + X[8,13] = 1
 onePed[9]: X[9,1] + X[9,2] + X[9,3] + X[9,4] + X[9,5] + X[9,6] + X[9,7]
   + X[9,8] + X[9,9] + X[9,10] + X[9,11] + X[9,12] + X[9,13] = 1
 onePed[10]: X[10,1] + X[10,2] + X[10,3] + X[10,4] + X[10,5] + X[10,6]
   + X[10,7] + X[10,8] + X[10,9] + X[10,10] + X[10,11] + X[10,12]
   + X[10,13] = 1
 onePed[11]: X[11,1] + X[11,2] + X[11,3] + X[11,4] + X[11,5] + X[11,6]
   + X[11,7] + X[11,8] + X[11,9] + X[11,10] + X[11,11] + X[11,12]
   + X[11,13] = 1
 onePed[12]: X[12,1] + X[12,2] + X[12,3] + X[12,4] + X[12,5] + X[12,6]
   + X[12,7] + X[12,8] + X[12,9] + X[12,10] + X[12,11] + X[12,12]
   + X[12,13] = 1
 onePed[13]: X[13,1] + X[13,2] + X[13,3] + X[13,4] + X[13,5] + X[13,6]
   + X[13,7] + X[13,8] + X[13,9] + X[13,10] + X[13,11] + X[13,12]
   + X[13,13] = 1
 onePed[14]: X[14,1] + X[14,2] + X[14,3] + X[14,4] + X[14,5] + X[14,6]
   + X[14,7] + X[14,8] + X[14,9] + X[14,10] + X[14,11] + X[14,12]
   + X[14,13] = 1
 onePed[15]: X[15,1] + X[15,2] + X[15,3] + X[15,4] + X[15,5] + X[15,6]
   + X[15,7] + X[15,8] + X[15,9] + X[15,10] + X[15,11] + X[15,12]
   + X[15,13] = 1
 onePed[16]: X[16,1] + X[16,2] + X[16,3] + X[16,4] + X[16,5] + X[16,6]
   + X[16,7] + X[16,8] + X[16,9] + X[16,10] + X[16,11] + X[16,12]
   + X[16,13] = 1
 onePed[17]: X[17,1] + X[17,2] + X[17,3] + X[17,4] + X[17,5] + X[17,6]
   + X[17,7] + X[17,8] + X[17,9] + X[17,10] + X[17,11] + X[17,12]
   + X[17,13] = 1
 onePed[18]: X[18,1] + X[18,2] + X[18,3] + X[18,4] + X[18,5] + X[18,6]
   + X[18,7] + X[18,8] + X[18,9] + X[18,10] + X[18,11] + X[18,12]
   + X[18,13] = 1
 onePed[19]: X[19,1] + X[19,2] + X[19,3] + X[19,4] + X[19,5] + X[19,6]
   + X[19,7] + X[19,8] + X[19,9] + X[19,10] + X[19,11] + X[19,12]
   + X[19,13] = 1
 onePed[20]: X[20,1] + X[20,2] + X[20,3] + X[20,4] + X[20,5] + X[20,6]
   + X[20,7] + X[20,8] + X[20,9] + X[20,10] + X[20,11] + X[20,12]
   + X[20,13] = 1
 onePed[21]: X[21,1] + X[21,2] + X[21,3] + X[21,4] + X[21,5] + X[21,6]
   + X[21,7] + X[21,8] + X[21,9] + X[21,10] + X[21,11] + X[21,12]
   + X[21,13] = 1
 onePed[22]: X[22,1] + X[22,2] + X[22,3] + X[22,4] + X[22,5] + X[22,6]
   + X[22,7] + X[22,8] + X[22,9] + X[22,10] + X[22,11] + X[22,12]
   + X[22,13] = 1
 onePed[23]: X[23,1] + X[23,2] + X[23,3] + X[23,4] + X[23,5] + X[23,6]
   + X[23,7] + X[23,8] + X[23,9] + X[23,10] + X[23,11] + X[23,12]
   + X[23,13] = 1
 onePed[24]: X[24,1] + X[24,2] + X[24,3] + X[24,4] + X[24,5] + X[24,6]
   + X[24,7] + X[24,8] + X[24,9] + X[24,10] + X[24,11] + X[24,12]
   + X[24,13] = 1
 onePed[25]: X[25,1] + X[25,2] + X[25,3] + X[25,4] + X[25,5] + X[25,6]
   + X[25,7] + X[25,8] + X[25,9] + X[25,10] + X[25,11] + X[25,12]
   + X[25,13] = 1
 onePed[26]: X[26,1] + X[26,2] + X[26,3] + X[26,4] + X[26,5] + X[26,6]
   + X[26,7] + X[26,8] + X[26,9] + X[26,10] + X[26,11] + X[26,12]
   + X[26,13] = 1
 onePed[27]: X[27,1] + X[27,2] + X[27,3] + X[27,4] + X[27,5] + X[27,6]
   + X[27,7] + X[27,8] + X[27,9] + X[27,10] + X[27,11] + X[27,12]
   + X[27,13] = 1
 onePed[28]: X[28,1] + X[28,2] + X[28,3] + X[28,4] + X[28,5] + X[28,6]
   + X[28,7] + X[28,8] + X[28,9] + X[28,10] + X[28,11] + X[28,12]
   + X[28,13] = 1
 onePed[29]: X[29,1] + X[29,2] + X[29,3] + X[29,4] + X[29,5] + X[29,6]
   + X[29,7] + X[29,8] + X[29,9] + X[29,10] + X[29,11] + X[29,12]
   + X[29,13] = 1
 onePed[30]: X[30,1] + X[30,2] + X[30,3] + X[30,4] + X[30,5] + X[30,6]
   + X[30,7] + X[30,8] + X[30,9] + X[30,10] + X[30,11] + X[30,12]
   + X[30,13] = 1
 onePed[31]: X[31,1] + X[31,2] + X[31,3] + X[31,4] + X[31,5] + X[31,6]
   + X[31,7] + X[31,8] + X[31,9] + X[31,10] + X[31,11] + X[31,12]
   + X[31,13] = 1
 onePed[32]: X[32,1] + X[32,2] + X[32,3] + X[32,4] + X[32,5] + X[32,6]
   + X[32,7] + X[32,8] + X[32,9] + X[32,10] + X[32,11] + X[32,12]
   + X[32,13] = 1
 onePed[33]: X[33,1] + X[33,2] + X[33,3] + X[33,4] + X[33,5] + X[33,6]
   + X[33,7] + X[33,8] + X[33,9] + X[33,10] + X[33,11] + X[33,12]
   + X[33,13] = 1
 onePed[34]: X[34,1] + X[34,2] + X[34,3] + X[34,4] + X[34,5] + X[34,6]
   + X[34,7] + X[34,8] + X[34,9] + X[34,10] + X[34,11] + X[34,12]
   + X[34,13] = 1
 onePed[35]: X[35,1] + X[35,2] + X[35,3] + X[35,4] + X[35,5] + X[35,6]
   + X[35,7] + X[35,8] + X[35,9] + X[35,10] + X[35,11] + X[35,12]
   + X[35,13] = 1
 onePed[36]: X[36,1] + X[36,2] + X[36,3] + X[36,4] + X[36,5] + X[36,6]
   + X[36,7] + X[36,8] + X[36,9] + X[36,10] + X[36,11] + X[36,12]
   + X[36,13] = 1
 onePed[37]: X[37,1] + X[37,2] + X[37,3] + X[37,4] + X[37,5] + X[37,6]
   + X[37,7] + X[37,8] + X[37,9] + X[37,10] + X[37,11] + X[37,12]
   + X[37,13] = 1
 onePed[38]: X[38,1] + X[38,2] + X[38,3] + X[38,4] + X[38,5] + X[38,6]
   + X[38,7] + X[38,8] + X[38,9] + X[38,10] + X[38,11] + X[38,12]
   + X[38,13] = 1
 onePed[39]: X[39,1] + X[39,2] + X[39,3] + X[39,4] + X[39,5] + X[39,6]
   + X[39,7] + X[39,8] + X[39,9] + X[39,10] + X[39,11] + X[39,12]
   + X[39,13] = 1
 onePed[40]: X[40,1] + X[40,2] + X[40,3] + X[40,4] + X[40,5] + X[40,6]
   + X[40,7] + X[40,8] + X[40,9] + X[40,10] + X[40,11] + X[40,12]
   + X[40,13] = 1
 onePed[41]: X[41,1] + X[41,2] + X[41,3] + X[41,4] + X[41,5] + X[41,6]
   + X[41,7] + X[41,8] + X[41,9] + X[41,10] + X[41,11] + X[41,12]
   + X[41,13] = 1
 onePed[42]: X[42,1] + X[42,2] + X[42,3] + X[42,4] + X[42,5] + X[42,6]
   + X[42,7] + X[42,8] + X[42,9] + X[42,10] + X[42,11] + X[42,12]
   + X[42,13] = 1
 expectedShifts[1]: X[1,1] + X[2,1] + X[3,1] + X[4,1] + X[5,1] + X[6,1]
   + X[7,1] + X[8,1] + X[9,1] + X[10,1] + X[11,1] + X[12,1] + X[13,1]
   + X[14,1] + X[15,1] + X[16,1] + X[17,1] + X[18,1] + X[19,1] + X[20,1]
   + X[21,1] + X[22,1] + X[23,1] + X[24,1] + X[25,1] + X[26,1] + X[27,1]
   + X[28,1] + X[29,1] + X[30,1] + X[31,1] + X[32,1] + X[33,1] + X[34,1]
   + X[35,1] + X[36,1] + X[37,1] + X[38,1] + X[39,1] + X[40,1] + X[41,1]
   + X[42,1] = 5
 expectedShifts[2]: X[1,2] + X[2,2] + X[3,2] + X[4,2] + X[5,2] + X[6,2]
   + X[7,2] + X[8,2] + X[9,2] + X[10,2] + X[11,2] + X[12,2] + X[13,2]
   + X[14,2] + X[15,2] + X[16,2] + X[17,2] + X[18,2] + X[19,2] + X[20,2]
   + X[21,2] + X[22,2] + X[23,2] + X[24,2] + X[25,2] + X[26,2] + X[27,2]
   + X[28,2] + X[29,2] + X[30,2] + X[31,2] + X[32,2] + X[33,2] + X[34,2]
   + X[35,2] + X[36,2] + X[37,2] + X[38,2] + X[39,2] + X[40,2] + X[41,2]
   + X[42,2] = 2
 expectedShifts[3]: X[1,3] + X[2,3] + X[3,3] + X[4,3] + X[5,3] + X[6,3]
   + X[7,3] + X[8,3] + X[9,3] + X[10,3] + X[11,3] + X[12,3] + X[13,3]
   + X[14,3] + X[15,3] + X[16,3] + X[17,3] + X[18,3] + X[19,3] + X[20,3]
   + X[21,3] + X[22,3] + X[23,3] + X[24,3] + X[25,3] + X[26,3] + X[27,3]
   + X[28,3] + X[29,3] + X[30,3] + X[31,3] + X[32,3] + X[33,3] + X[34,3]
   + X[35,3] + X[36,3] + X[37,3] + X[38,3] + X[39,3] + X[40,3] + X[41,3]
   + X[42,3] = 4
 expectedShifts[4]: X[1,4] + X[2,4] + X[3,4] + X[4,4] + X[5,4] + X[6,4]
   + X[7,4] + X[8,4] + X[9,4] + X[10,4] + X[11,4] + X[12,4] + X[13,4]
   + X[14,4] + X[15,4] + X[16,4] + X[17,4] + X[18,4] + X[19,4] + X[20,4]
   + X[21,4] + X[22,4] + X[23,4] + X[24,4] + X[25,4] + X[26,4] + X[27,4]
   + X[28,4] + X[29,4] + X[30,4] + X[31,4] + X[32,4] + X[33,4] + X[34,4]
   + X[35,4] + X[36,4] + X[37,4] + X[38,4] + X[39,4] + X[40,4] + X[41,4]
   + X[42,4] = 3
 expectedShifts[5]: X[1,5] + X[2,5] + X[3,5] + X[4,5] + X[5,5] + X[6,5]
   + X[7,5] + X[8,5] + X[9,5] + X[10,5] + X[11,5] + X[12,5] + X[13,5]
   + X[14,5] + X[15,5] + X[16,5] + X[17,5] + X[18,5] + X[19,5] + X[20,5]
   + X[21,5] + X[22,5] + X[23,5] + X[24,5] + X[25,5] + X[26,5] + X[27,5]
   + X[28,5] + X[29,5] + X[30,5] + X[31,5] + X[32,5] + X[33,5] + X[34,5]
   + X[35,5] + X[36,5] + X[37,5] + X[38,5] + X[39,5] + X[40,5] + X[41,5]
   + X[42,5] = 3
 expectedShifts[6]: X[1,6] + X[2,6] + X[3,6] + X[4,6] + X[5,6] + X[6,6]
   + X[7,6] + X[8,6] + X[9,6] + X[10,6] + X[11,6] + X[12,6] + X[13,6]
   + X[14,6] + X[15,6] + X[16,6] + X[17,6] + X[18,6] + X[19,6] + X[20,6]
   + X[21,6] + X[22,6] + X[23,6] + X[24,6] + X[25,6] + X[26,6] + X[27,6]
   + X[28,6] + X[29,6] + X[30,6] + X[31,6] + X[32,6] + X[33,6] + X[34,6]
   + X[35,6] + X[36,6] + X[37,6] + X[38,6] + X[39,6] + X[40,6] + X[41,6]
   + X[42,6] = 2
 expectedShifts[7]: X[1,7] + X[2,7] + X[3,7] + X[4,7] + X[5,7] + X[6,7]
   + X[7,7] + X[8,7] + X[9,7] + X[10,7] + X[11,7] + X[12,7] + X[13,7]
   + X[14,7] + X[15,7] + X[16,7] + X[17,7] + X[18,7] + X[19,7] + X[20,7]
   + X[21,7] + X[22,7] + X[23,7] + X[24,7] + X[25,7] + X[26,7] + X[27,7]
   + X[28,7] + X[29,7] + X[30,7] + X[31,7] + X[32,7] + X[33,7] + X[34,7]
   + X[35,7] + X[36,7] + X[37,7] + X[38,7] + X[39,7] + X[40,7] + X[41,7]
   + X[42,7] = 3
 expectedShifts[8]: X[1,8] + X[2,8] + X[3,8] + X[4,8] + X[5,8] + X[6,8]
   + X[7,8] + X[8,8] + X[9,8] + X[10,8] + X[11,8] + X[12,8] + X[13,8]
   + X[14,8] + X[15,8] + X[16,8] + X[17,8] + X[18,8] + X[19,8] + X[20,8]
   + X[21,8] + X[22,8] + X[23,8] + X[24,8] + X[25,8] + X[26,8] + X[27,8]
   + X[28,8] + X[29,8] + X[30,8] + X[31,8] + X[32,8] + X[33,8] + X[34,8]
   + X[35,8] + X[36,8] + X[37,8] + X[38,8] + X[39,8] + X[40,8] + X[41,8]
   + X[42,8] = 3
 expectedShifts[9]: X[1,9] + X[2,9] + X[3,9] + X[4,9] + X[5,9] + X[6,9]
   + X[7,9] + X[8,9] + X[9,9] + X[10,9] + X[11,9] + X[12,9] + X[13,9]
   + X[14,9] + X[15,9] + X[16,9] + X[17,9] + X[18,9] + X[19,9] + X[20,9]
   + X[21,9] + X[22,9] + X[23,9] + X[24,9] + X[25,9] + X[26,9] + X[27,9]
   + X[28,9] + X[29,9] + X[30,9] + X[31,9] + X[32,9] + X[33,9] + X[34,9]
   + X[35,9] + X[36,9] + X[37,9] + X[38,9] + X[39,9] + X[40,9] + X[41,9]
   + X[42,9] = 5
 expectedShifts[10]: X[1,10] + X[2,10] + X[3,10] + X[4,10] + X[5,10]
   + X[6,10] + X[7,10] + X[8,10] + X[9,10] + X[10,10] + X[11,10] + X[12,10]
   + X[13,10] + X[14,10] + X[15,10] + X[16,10] + X[17,10] + X[18,10]
   + X[19,10] + X[20,10] + X[21,10] + X[22,10] + X[23,10] + X[24,10]
   + X[25,10] + X[26,10] + X[27,10] + X[28,10] + X[29,10] + X[30,10]
   + X[31,10] + X[32,10] + X[33,10] + X[34,10] + X[35,10] + X[36,10]
   + X[37,10] + X[38,10] + X[39,10] + X[40,10] + X[41,10] + X[42,10] = 3
 expectedShifts[11]: X[1,11] + X[2,11] + X[3,11] + X[4,11] + X[5,11]
   + X[6,11] + X[7,11] + X[8,11] + X[9,11] + X[10,11] + X[11,11] + X[12,11]
   + X[13,11] + X[14,11] + X[15,11] + X[16,11] + X[17,11] + X[18,11]
   + X[19,11] + X[20,11] + X[21,11] + X[22,11] + X[23,11] + X[24,11]
   + X[25,11] + X[26,11] + X[27,11] + X[28,11] + X[29,11] + X[30,11]
   + X[31,11] + X[32,11] + X[33,11] + X[34,11] + X[35,11] + X[36,11]
   + X[37,11] + X[38,11] + X[39,11] + X[40,11] + X[41,11] + X[42,11] = 4
 expectedShifts[12]: X[1,12] + X[2,12] + X[3,12] + X[4,12] + X[5,12]
   + X[6,12] + X[7,12] + X[8,12] + X[9,12] + X[10,12] + X[11,12] + X[12,12]
   + X[13,12] + X[14,12] + X[15,12] + X[16,12] + X[17,12] + X[18,12]
   + X[19,12] + X[20,12] + X[21,12] + X[22,12] + X[23,12] + X[24,12]
   + X[25,12] + X[26,12] + X[27,12] + X[28,12] + X[29,12] + X[30,12]
   + X[31,12] + X[32,12] + X[33,12] + X[34,12] + X[35,12] + X[36,12]
   + X[37,12] + X[38,12] + X[39,12] + X[40,12] + X[41,12] + X[42,12] = 2
 expectedShifts[13]: X[1,13] + X[2,13] + X[3,13] + X[4,13] + X[5,13]
   + X[6,13] + X[7,13] + X[8,13] + X[9,13] + X[10,13] + X[11,13] + X[12,13]
   + X[13,13] + X[14,13] + X[15,13] + X[16,13] + X[17,13] + X[18,13]
   + X[19,13] + X[20,13] + X[21,13] + X[22,13] + X[23,13] + X[24,13]
   + X[25,13] + X[26,13] + X[27,13] + X[28,13] + X[29,13] + X[30,13]
   + X[31,13] + X[32,13] + X[33,13] + X[34,13] + X[35,13] + X[36,13]
   + X[37,13] + X[38,13] + X[39,13] + X[40,13] + X[41,13] + X[42,13] = 3
 availability[1,1]: X[1,1] <= 0
 availability[1,2]: X[1,2] <= 0
 availability[1,3]: X[1,3] <= 0
 availability[1,4]: X[1,4] <= 1
 availability[1,5]: X[1,5] <= 1
 availability[1,6]: X[1,6] <= 1
 availability[1,7]: X[1,7] <= 1
 availability[1,8]: X[1,8] <= 1
 availability[1,9]: X[1,9] <= 1
 availability[1,10]: X[1,10] <= 1
 availability[1,11]: X[1,11] <= 1
 availability[1,12]: X[1,12] <= 1
 availability[1,13]: X[1,13] <= 1
 availability[2,1]: X[2,1] <= 0
 availability[2,2]: X[2,2] <= 0
 availability[2,3]: X[2,3] <= 0
 availability[2,4]: X[2,4] <= 1
 availability[2,5]: X[2,5] <= 1
 availability[2,6]: X[2,6] <= 1
 availability[2,7]: X[2,7] <= 1
 availability[2,8]: X[2,8] <= 1
 availability[2,9]: X[2,9] <= 1
 availability[2,10]: X[2,10] <= 1
 availability[2,11]: X[2,11] <= 1
 availability[2,12]: X[2,12] <= 1
 availability[2,13]: X[2,13] <= 1
 availability[3,1]: X[3,1] <= 1
 availability[3,2]: X[3,2] <= 0
 availability[3,3]: X[3,3] <= 0
 availability[3,4]: X[3,4] <= 1
 availability[3,5]: X[3,5] <= 1
 availability[3,6]: X[3,6] <= 1
 availability[3,7]: X[3,7] <= 1
 availability[3,8]: X[3,8] <= 1
 availability[3,9]: X[3,9] <= 1
 availability[3,10]: X[3,10] <= 1
 availability[3,11]: X[3,11] <= 1
 availability[3,12]: X[3,12] <= 1
 availability[3,13]: X[3,13] <= 0
 availability[4,1]: X[4,1] <= 1
 availability[4,2]: X[4,2] <= 0
 availability[4,3]: X[4,3] <= 0
 availability[4,4]: X[4,4] <= 1
 availability[4,5]: X[4,5] <= 1
 availability[4,6]: X[4,6] <= 1
 availability[4,7]: X[4,7] <= 1
 availability[4,8]: X[4,8] <= 1
 availability[4,9]: X[4,9] <= 1
 availability[4,10]: X[4,10] <= 1
 availability[4,11]: X[4,11] <= 1
 availability[4,12]: X[4,12] <= 1
 availability[4,13]: X[4,13] <= 0
 availability[5,1]: X[5,1] <= 1
 availability[5,2]: X[5,2] <= 0
 availability[5,3]: X[5,3] <= 1
 availability[5,4]: X[5,4] <= 1
 availability[5,5]: X[5,5] <= 1
 availability[5,6]: X[5,6] <= 1
 availability[5,7]: X[5,7] <= 0
 availability[5,8]: X[5,8] <= 1
 availability[5,9]: X[5,9] <= 1
 availability[5,10]: X[5,10] <= 1
 availability[5,11]: X[5,11] <= 1
 availability[5,12]: X[5,12] <= 1
 availability[5,13]: X[5,13] <= 1
 availability[6,1]: X[6,1] <= 1
 availability[6,2]: X[6,2] <= 0
 availability[6,3]: X[6,3] <= 1
 availability[6,4]: X[6,4] <= 1
 availability[6,5]: X[6,5] <= 1
 availability[6,6]: X[6,6] <= 1
 availability[6,7]: X[6,7] <= 0
 availability[6,8]: X[6,8] <= 1
 availability[6,9]: X[6,9] <= 1
 availability[6,10]: X[6,10] <= 1
 availability[6,11]: X[6,11] <= 1
 availability[6,12]: X[6,12] <= 1
 availability[6,13]: X[6,13] <= 1
 availability[7,1]: X[7,1] <= 1
 availability[7,2]: X[7,2] <= 0
 availability[7,3]: X[7,3] <= 1
 availability[7,4]: X[7,4] <= 1
 availability[7,5]: X[7,5] <= 1
 availability[7,6]: X[7,6] <= 1
 availability[7,7]: X[7,7] <= 0
 availability[7,8]: X[7,8] <= 1
 availability[7,9]: X[7,9] <= 1
 availability[7,10]: X[7,10] <= 1
 availability[7,11]: X[7,11] <= 1
 availability[7,12]: X[7,12] <= 1
 availability[7,13]: X[7,13] <= 1
 availability[8,1]: X[8,1] <= 1
 availability[8,2]: X[8,2] <= 0
 availability[8,3]: X[8,3] <= 1
 availability[8,4]: X[8,4] <= 1
 availability[8,5]: X[8,5] <= 1
 availability[8,6]: X[8,6] <= 1
 availability[8,7]: X[8,7] <= 0
 availability[8,8]: X[8,8] <= 1
 availability[8,9]: X[8,9] <= 1
 availability[8,10]: X[8,10] <= 1
 availability[8,11]: X[8,11] <= 1
 availability[8,12]: X[8,12] <= 1
 availability[8,13]: X[8,13] <= 1
 availability[9,1]: X[9,1] <= 1
 availability[9,2]: X[9,2] <= 0
 availability[9,3]: X[9,3] <= 1
 availability[9,4]: X[9,4] <= 1
 availability[9,5]: X[9,5] <= 1
 availability[9,6]: X[9,6] <= 1
 availability[9,7]: X[9,7] <= 1
 availability[9,8]: X[9,8] <= 1
 availability[9,9]: X[9,9] <= 1
 availability[9,10]: X[9,10] <= 0
 availability[9,11]: X[9,11] <= 1
 availability[9,12]: X[9,12] <= 0
 availability[9,13]: X[9,13] <= 1
 availability[10,1]: X[10,1] <= 1
 availability[10,2]: X[10,2] <= 1
 availability[10,3]: X[10,3] <= 1
 availability[10,4]: X[10,4] <= 1
 availability[10,5]: X[10,5] <= 1
 availability[10,6]: X[10,6] <= 1
 availability[10,7]: X[10,7] <= 1
 availability[10,8]: X[10,8] <= 1
 availability[10,9]: X[10,9] <= 1
 availability[10,10]: X[10,10] <= 0
 availability[10,11]: X[10,11] <= 1
 availability[10,12]: X[10,12] <= 0
 availability[10,13]: X[10,13] <= 1
 availability[11,1]: X[11,1] <= 1
 availability[11,2]: X[11,2] <= 1
 availability[11,3]: X[11,3] <= 1
 availability[11,4]: X[11,4] <= 1
 availability[11,5]: X[11,5] <= 1
 availability[11,6]: X[11,6] <= 1
 availability[11,7]: X[11,7] <= 1
 availability[11,8]: X[11,8] <= 1
 availability[11,9]: X[11,9] <= 1
 availability[11,10]: X[11,10] <= 0
 availability[11,11]: X[11,11] <= 1
 availability[11,12]: X[11,12] <= 0
 availability[11,13]: X[11,13] <= 1
 availability[12,1]: X[12,1] <= 1
 availability[12,2]: X[12,2] <= 1
 availability[12,3]: X[12,3] <= 1
 availability[12,4]: X[12,4] <= 1
 availability[12,5]: X[12,5] <= 1
 availability[12,6]: X[12,6] <= 1
 availability[12,7]: X[12,7] <= 1
 availability[12,8]: X[12,8] <= 1
 availability[12,9]: X[12,9] <= 1
 availability[12,10]: X[12,10] <= 1
 availability[12,11]: X[12,11] <= 1
 availability[12,12]: X[12,12] <= 0
 availability[12,13]: X[12,13] <= 1
 availability[13,1]: X[13,1] <= 1
 availability[13,2]: X[13,2] <= 1
 availability[13,3]: X[13,3] <= 1
 availability[13,4]: X[13,4] <= 1
 availability[13,5]: X[13,5] <= 1
 availability[13,6]: X[13,6] <= 1
 availability[13,7]: X[13,7] <= 1
 availability[13,8]: X[13,8] <= 1
 availability[13,9]: X[13,9] <= 1
 availability[13,10]: X[13,10] <= 1
 availability[13,11]: X[13,11] <= 1
 availability[13,12]: X[13,12] <= 1
 availability[13,13]: X[13,13] <= 1
 availability[14,1]: X[14,1] <= 1
 availability[14,2]: X[14,2] <= 1
 availability[14,3]: X[14,3] <= 1
 availability[14,4]: X[14,4] <= 1
 availability[14,5]: X[14,5] <= 1
 availability[14,6]: X[14,6] <= 1
 availability[14,7]: X[14,7] <= 1
 availability[14,8]: X[14,8] <= 1
 availability[14,9]: X[14,9] <= 1
 availability[14,10]: X[14,10] <= 1
 availability[14,11]: X[14,11] <= 1
 availability[14,12]: X[14,12] <= 1
 availability[14,13]: X[14,13] <= 1
 availability[15,1]: X[15,1] <= 0
 availability[15,2]: X[15,2] <= 0
 availability[15,3]: X[15,3] <= 1
 availability[15,4]: X[15,4] <= 1
 availability[15,5]: X[15,5] <= 1
 availability[15,6]: X[15,6] <= 1
 availability[15,7]: X[15,7] <= 1
 availability[15,8]: X[15,8] <= 1
 availability[15,9]: X[15,9] <= 1
 availability[15,10]: X[15,10] <= 1
 availability[15,11]: X[15,11] <= 1
 availability[15,12]: X[15,12] <= 1
 availability[15,13]: X[15,13] <= 0
 availability[16,1]: X[16,1] <= 0
 availability[16,2]: X[16,2] <= 0
 availability[16,3]: X[16,3] <= 1
 availability[16,4]: X[16,4] <= 1
 availability[16,5]: X[16,5] <= 1
 availability[16,6]: X[16,6] <= 1
 availability[16,7]: X[16,7] <= 1
 availability[16,8]: X[16,8] <= 1
 availability[16,9]: X[16,9] <= 1
 availability[16,10]: X[16,10] <= 1
 availability[16,11]: X[16,11] <= 1
 availability[16,12]: X[16,12] <= 1
 availability[16,13]: X[16,13] <= 0
 availability[17,1]: X[17,1] <= 0
 availability[17,2]: X[17,2] <= 0
 availability[17,3]: X[17,3] <= 1
 availability[17,4]: X[17,4] <= 1
 availability[17,5]: X[17,5] <= 1
 availability[17,6]: X[17,6] <= 1
 availability[17,7]: X[17,7] <= 1
 availability[17,8]: X[17,8] <= 1
 availability[17,9]: X[17,9] <= 1
 availability[17,10]: X[17,10] <= 1
 availability[17,11]: X[17,11] <= 1
 availability[17,12]: X[17,12] <= 1
 availability[17,13]: X[17,13] <= 0
 availability[18,1]: X[18,1] <= 0
 availability[18,2]: X[18,2] <= 0
 availability[18,3]: X[18,3] <= 1
 availability[18,4]: X[18,4] <= 1
 availability[18,5]: X[18,5] <= 1
 availability[18,6]: X[18,6] <= 1
 availability[18,7]: X[18,7] <= 1
 availability[18,8]: X[18,8] <= 1
 availability[18,9]: X[18,9] <= 1
 availability[18,10]: X[18,10] <= 1
 availability[18,11]: X[18,11] <= 1
 availability[18,12]: X[18,12] <= 1
 availability[18,13]: X[18,13] <= 0
 availability[19,1]: X[19,1] <= 0
 availability[19,2]: X[19,2] <= 0
 availability[19,3]: X[19,3] <= 1
 availability[19,4]: X[19,4] <= 0
 availability[19,5]: X[19,5] <= 1
 availability[19,6]: X[19,6] <= 1
 availability[19,7]: X[19,7] <= 1
 availability[19,8]: X[19,8] <= 1
 availability[19,9]: X[19,9] <= 1
 availability[19,10]: X[19,10] <= 1
 availability[19,11]: X[19,11] <= 1
 availability[19,12]: X[19,12] <= 1
 availability[19,13]: X[19,13] <= 1
 availability[20,1]: X[20,1] <= 0
 availability[20,2]: X[20,2] <= 0
 availability[20,3]: X[20,3] <= 1
 availability[20,4]: X[20,4] <= 0
 availability[20,5]: X[20,5] <= 1
 availability[20,6]: X[20,6] <= 1
 availability[20,7]: X[20,7] <= 1
 availability[20,8]: X[20,8] <= 1
 availability[20,9]: X[20,9] <= 1
 availability[20,10]: X[20,10] <= 1
 availability[20,11]: X[20,11] <= 1
 availability[20,12]: X[20,12] <= 1
 availability[20,13]: X[20,13] <= 1
 availability[21,1]: X[21,1] <= 0
 availability[21,2]: X[21,2] <= 0
 availability[21,3]: X[21,3] <= 1
 availability[21,4]: X[21,4] <= 0
 availability[21,5]: X[21,5] <= 1
 availability[21,6]: X[21,6] <= 1
 availability[21,7]: X[21,7] <= 1
 availability[21,8]: X[21,8] <= 1
 availability[21,9]: X[21,9] <= 1
 availability[21,10]: X[21,10] <= 1
 availability[21,11]: X[21,11] <= 1
 availability[21,12]: X[21,12] <= 1
 availability[21,13]: X[21,13] <= 1
 availability[22,1]: X[22,1] <= 0
 availability[22,2]: X[22,2] <= 0
 availability[22,3]: X[22,3] <= 1
 availability[22,4]: X[22,4] <= 0
 availability[22,5]: X[22,5] <= 1
 availability[22,6]: X[22,6] <= 1
 availability[22,7]: X[22,7] <= 1
 availability[22,8]: X[22,8] <= 1
 availability[22,9]: X[22,9] <= 1
 availability[22,10]: X[22,10] <= 1
 availability[22,11]: X[22,11] <= 1
 availability[22,12]: X[22,12] <= 1
 availability[22,13]: X[22,13] <= 1
 availability[23,1]: X[23,1] <= 1
 availability[23,2]: X[23,2] <= 0
 availability[23,3]: X[23,3] <= 1
 availability[23,4]: X[23,4] <= 1
 availability[23,5]: X[23,5] <= 1
 availability[23,6]: X[23,6] <= 1
 availability[23,7]: X[23,7] <= 0
 availability[23,8]: X[23,8] <= 1
 availability[23,9]: X[23,9] <= 1
 availability[23,10]: X[23,10] <= 1
 availability[23,11]: X[23,11] <= 0
 availability[23,12]: X[23,12] <= 1
 availability[23,13]: X[23,13] <= 1
 availability[24,1]: X[24,1] <= 1
 availability[24,2]: X[24,2] <= 1
 availability[24,3]: X[24,3] <= 1
 availability[24,4]: X[24,4] <= 1
 availability[24,5]: X[24,5] <= 1
 availability[24,6]: X[24,6] <= 1
 availability[24,7]: X[24,7] <= 0
 availability[24,8]: X[24,8] <= 1
 availability[24,9]: X[24,9] <= 1
 availability[24,10]: X[24,10] <= 1
 availability[24,11]: X[24,11] <= 0
 availability[24,12]: X[24,12] <= 1
 availability[24,13]: X[24,13] <= 1
 availability[25,1]: X[25,1] <= 1
 availability[25,2]: X[25,2] <= 1
 availability[25,3]: X[25,3] <= 1
 availability[25,4]: X[25,4] <= 1
 availability[25,5]: X[25,5] <= 1
 availability[25,6]: X[25,6] <= 1
 availability[25,7]: X[25,7] <= 0
 availability[25,8]: X[25,8] <= 1
 availability[25,9]: X[25,9] <= 1
 availability[25,10]: X[25,10] <= 1
 availability[25,11]: X[25,11] <= 0
 availability[25,12]: X[25,12] <= 1
 availability[25,13]: X[25,13] <= 1
 availability[26,1]: X[26,1] <= 1
 availability[26,2]: X[26,2] <= 1
 availability[26,3]: X[26,3] <= 1
 availability[26,4]: X[26,4] <= 1
 availability[26,5]: X[26,5] <= 1
 availability[26,6]: X[26,6] <= 1
 availability[26,7]: X[26,7] <= 0
 availability[26,8]: X[26,8] <= 1
 availability[26,9]: X[26,9] <= 1
 availability[26,10]: X[26,10] <= 1
 availability[26,11]: X[26,11] <= 0
 availability[26,12]: X[26,12] <= 1
 availability[26,13]: X[26,13] <= 1
 availability[27,1]: X[27,1] <= 1
 availability[27,2]: X[27,2] <= 1
 availability[27,3]: X[27,3] <= 1
 availability[27,4]: X[27,4] <= 1
 availability[27,5]: X[27,5] <= 1
 availability[27,6]: X[27,6] <= 1
 availability[27,7]: X[27,7] <= 1
 availability[27,8]: X[27,8] <= 1
 availability[27,9]: X[27,9] <= 1
 availability[27,10]: X[27,10] <= 1
 availability[27,11]: X[27,11] <= 0
 availability[27,12]: X[27,12] <= 1
 availability[27,13]: X[27,13] <= 1
 availability[28,1]: X[28,1] <= 0
 availability[28,2]: X[28,2] <= 1
 availability[28,3]: X[28,3] <= 1
 availability[28,4]: X[28,4] <= 1
 availability[28,5]: X[28,5] <= 1
 availability[28,6]: X[28,6] <= 1
 availability[28,7]: X[28,7] <= 1
 availability[28,8]: X[28,8] <= 1
 availability[28,9]: X[28,9] <= 1
 availability[28,10]: X[28,10] <= 1
 availability[28,11]: X[28,11] <= 0
 availability[28,12]: X[28,12] <= 1
 availability[28,13]: X[28,13] <= 1
 availability[29,1]: X[29,1] <= 1
 availability[29,2]: X[29,2] <= 0
 availability[29,3]: X[29,3] <= 0
 availability[29,4]: X[29,4] <= 1
 availability[29,5]: X[29,5] <= 0
 availability[29,6]: X[29,6] <= 1
 availability[29,7]: X[29,7] <= 1
 availability[29,8]: X[29,8] <= 1
 availability[29,9]: X[29,9] <= 1
 availability[29,10]: X[29,10] <= 1
 availability[29,11]: X[29,11] <= 1
 availability[29,12]: X[29,12] <= 1
 availability[29,13]: X[29,13] <= 1
 availability[30,1]: X[30,1] <= 1
 availability[30,2]: X[30,2] <= 0
 availability[30,3]: X[30,3] <= 1
 availability[30,4]: X[30,4] <= 1
 availability[30,5]: X[30,5] <= 0
 availability[30,6]: X[30,6] <= 1
 availability[30,7]: X[30,7] <= 1
 availability[30,8]: X[30,8] <= 1
 availability[30,9]: X[30,9] <= 1
 availability[30,10]: X[30,10] <= 1
 availability[30,11]: X[30,11] <= 1
 availability[30,12]: X[30,12] <= 1
 availability[30,13]: X[30,13] <= 1
 availability[31,1]: X[31,1] <= 1
 availability[31,2]: X[31,2] <= 0
 availability[31,3]: X[31,3] <= 1
 availability[31,4]: X[31,4] <= 1
 availability[31,5]: X[31,5] <= 0
 availability[31,6]: X[31,6] <= 1
 availability[31,7]: X[31,7] <= 1
 availability[31,8]: X[31,8] <= 1
 availability[31,9]: X[31,9] <= 1
 availability[31,10]: X[31,10] <= 1
 availability[31,11]: X[31,11] <= 1
 availability[31,12]: X[31,12] <= 0
 availability[31,13]: X[31,13] <= 1
 availability[32,1]: X[32,1] <= 1
 availability[32,2]: X[32,2] <= 0
 availability[32,3]: X[32,3] <= 1
 availability[32,4]: X[32,4] <= 1
 availability[32,5]: X[32,5] <= 0
 availability[32,6]: X[32,6] <= 1
 availability[32,7]: X[32,7] <= 1
 availability[32,8]: X[32,8] <= 1
 availability[32,9]: X[32,9] <= 1
 availability[32,10]: X[32,10] <= 1
 availability[32,11]: X[32,11] <= 1
 availability[32,12]: X[32,12] <= 0
 availability[32,13]: X[32,13] <= 1
 availability[33,1]: X[33,1] <= 1
 availability[33,2]: X[33,2] <= 0
 availability[33,3]: X[33,3] <= 1
 availability[33,4]: X[33,4] <= 1
 availability[33,5]: X[33,5] <= 1
 availability[33,6]: X[33,6] <= 1
 availability[33,7]: X[33,7] <= 1
 availability[33,8]: X[33,8] <= 1
 availability[33,9]: X[33,9] <= 1
 availability[33,10]: X[33,10] <= 1
 availability[33,11]: X[33,11] <= 1
 availability[33,12]: X[33,12] <= 1
 availability[33,13]: X[33,13] <= 1
 availability[34,1]: X[34,1] <= 1
 availability[34,2]: X[34,2] <= 0
 availability[34,3]: X[34,3] <= 1
 availability[34,4]: X[34,4] <= 1
 availability[34,5]: X[34,5] <= 1
 availability[34,6]: X[34,6] <= 1
 availability[34,7]: X[34,7] <= 1
 availability[34,8]: X[34,8] <= 1
 availability[34,9]: X[34,9] <= 1
 availability[34,10]: X[34,10] <= 1
 availability[34,11]: X[34,11] <= 1
 availability[34,12]: X[34,12] <= 1
 availability[34,13]: X[34,13] <= 1
 availability[35,1]: X[35,1] <= 1
 availability[35,2]: X[35,2] <= 0
 availability[35,3]: X[35,3] <= 1
 availability[35,4]: X[35,4] <= 1
 availability[35,5]: X[35,5] <= 1
 availability[35,6]: X[35,6] <= 1
 availability[35,7]: X[35,7] <= 1
 availability[35,8]: X[35,8] <= 1
 availability[35,9]: X[35,9] <= 1
 availability[35,10]: X[35,10] <= 1
 availability[35,11]: X[35,11] <= 1
 availability[35,12]: X[35,12] <= 1
 availability[35,13]: X[35,13] <= 1
 availability[36,1]: X[36,1] <= 1
 availability[36,2]: X[36,2] <= 0
 availability[36,3]: X[36,3] <= 1
 availability[36,4]: X[36,4] <= 1
 availability[36,5]: X[36,5] <= 1
 availability[36,6]: X[36,6] <= 1
 availability[36,7]: X[36,7] <= 1
 availability[36,8]: X[36,8] <= 1
 availability[36,9]: X[36,9] <= 1
 availability[36,10]: X[36,10] <= 1
 availability[36,11]: X[36,11] <= 1
 availability[36,12]: X[36,12] <= 1
 availability[36,13]: X[36,13] <= 1
 availability[37,1]: X[37,1] <= 1
 availability[37,2]: X[37,2] <= 0
 availability[37,3]: X[37,3] <= 1
 availability[37,4]: X[37,4] <= 1
 availability[37,5]: X[37,5] <= 1
 availability[37,6]: X[37,6] <= 1
 availability[37,7]: X[37,7] <= 1
 availability[37,8]: X[37,8] <= 1
 availability[37,9]: X[37,9] <= 1
 availability[37,10]: X[37,10] <= 1
 availability[37,11]: X[37,11] <= 1
 availability[37,12]: X[37,12] <= 1
 availability[37,13]: X[37,13] <= 1
 availability[38,1]: X[38,1] <= 1
 availability[38,2]: X[38,2] <= 0
 availability[38,3]: X[38,3] <= 1
 availability[38,4]: X[38,4] <= 1
 availability[38,5]: X[38,5] <= 1
 availability[38,6]: X[38,6] <= 1
 availability[38,7]: X[38,7] <= 1
 availability[38,8]: X[38,8] <= 1
 availability[38,9]: X[38,9] <= 1
 availability[38,10]: X[38,10] <= 1
 availability[38,11]: X[38,11] <= 1
 availability[38,12]: X[38,12] <= 1
 availability[38,13]: X[38,13] <= 1
 availability[39,1]: X[39,1] <= 1
 availability[39,2]: X[39,2] <= 0
 availability[39,3]: X[39,3] <= 1
 availability[39,4]: X[39,4] <= 0
 availability[39,5]: X[39,5] <= 1
 availability[39,6]: X[39,6] <= 1
 availability[39,7]: X[39,7] <= 1
 availability[39,8]: X[39,8] <= 1
 availability[39,9]: X[39,9] <= 1
 availability[39,10]: X[39,10] <= 1
 availability[39,11]: X[39,11] <= 1
 availability[39,12]: X[39,12] <= 1
 availability[39,13]: X[39,13] <= 1
 availability[40,1]: X[40,1] <= 1
 availability[40,2]: X[40,2] <= 0
 availability[40,3]: X[40,3] <= 1
 availability[40,4]: X[40,4] <= 0
 availability[40,5]: X[40,5] <= 1
 availability[40,6]: X[40,6] <= 1
 availability[40,7]: X[40,7] <= 1
 availability[40,8]: X[40,8] <= 1
 availability[40,9]: X[40,9] <= 1
 availability[40,10]: X[40,10] <= 1
 availability[40,11]: X[40,11] <= 1
 availability[40,12]: X[40,12] <= 1
 availability[40,13]: X[40,13] <= 1
 availability[41,1]: X[41,1] <= 1
 availability[41,2]: X[41,2] <= 0
 availability[41,3]: X[41,3] <= 1
 availability[41,4]: X[41,4] <= 0
 availability[41,5]: X[41,5] <= 1
 availability[41,6]: X[41,6] <= 1
 availability[41,7]: X[41,7] <= 1
 availability[41,8]: X[41,8] <= 1
 availability[41,9]: X[41,9] <= 1
 availability[41,10]: X[41,10] <= 1
 availability[41,11]: X[41,11] <= 1
 availability[41,12]: X[41,12] <= 1
 availability[41,13]: X[41,13] <= 1
 availability[42,1]: X[42,1] <= 1
 availability[42,2]: X[42,2] <= 0
 availability[42,3]: X[42,3] <= 1
 availability[42,4]: X[42,4] <= 0
 availability[42,5]: X[42,5] <= 1
 availability[42,6]: X[42,6] <= 1
 availability[42,7]: X[42,7] <= 1
 availability[42,8]: X[42,8] <= 1
 availability[42,9]: X[42,9] <= 1
 availability[42,10]: X[42,10] <= 1
 availability[42,11]: X[42,11] <= 1
 availability[42,12]: X[42,12] <= 1
 availability[42,13]: X[42,13] <= 1
 noBackToBack[1,1]: X[1,1] + X[2,1] <= 1
 noBackToBack[1,2]: X[1,2] + X[2,2] <= 1
 noBackToBack[1,3]: X[1,3] + X[2,3] <= 1
 noBackToBack[1,4]: X[1,4] + X[2,4] <= 1
 noBackToBack[1,5]: X[1,5] + X[2,5] <= 1
 noBackToBack[1,6]: X[1,6] + X[2,6] <= 1
 noBackToBack[1,7]: X[1,7] + X[2,7] <= 1
 noBackToBack[1,8]: X[1,8] + X[2,8] <= 1
 noBackToBack[1,9]: X[1,9] + X[2,9] <= 1
 noBackToBack[1,10]: X[1,10] + X[2,10] <= 1
 noBackToBack[1,11]: X[1,11] + X[2,11] <= 1
 noBackToBack[1,12]: X[1,12] + X[2,12] <= 1
 noBackToBack[1,13]: X[1,13] + X[2,13] <= 1
 noBackToBack[2,1]: X[2,1] + X[3,1] <= 1
 noBackToBack[2,2]: X[2,2] + X[3,2] <= 1
 noBackToBack[2,3]: X[2,3] + X[3,3] <= 1
 noBackToBack[2,4]: X[2,4] + X[3,4] <= 1
 noBackToBack[2,5]: X[2,5] + X[3,5] <= 1
 noBackToBack[2,6]: X[2,6] + X[3,6] <= 1
 noBackToBack[2,7]: X[2,7] + X[3,7] <= 1
 noBackToBack[2,8]: X[2,8] + X[3,8] <= 1
 noBackToBack[2,9]: X[2,9] + X[3,9] <= 1
 noBackToBack[2,10]: X[2,10] + X[3,10] <= 1
 noBackToBack[2,11]: X[2,11] + X[3,11] <= 1
 noBackToBack[2,12]: X[2,12] + X[3,12] <= 1
 noBackToBack[2,13]: X[2,13] + X[3,13] <= 1
 noBackToBack[3,1]: X[3,1] + X[4,1] <= 1
 noBackToBack[3,2]: X[3,2] + X[4,2] <= 1
 noBackToBack[3,3]: X[3,3] + X[4,3] <= 1
 noBackToBack[3,4]: X[3,4] + X[4,4] <= 1
 noBackToBack[3,5]: X[3,5] + X[4,5] <= 1
 noBackToBack[3,6]: X[3,6] + X[4,6] <= 1
 noBackToBack[3,7]: X[3,7] + X[4,7] <= 1
 noBackToBack[3,8]: X[3,8] + X[4,8] <= 1
 noBackToBack[3,9]: X[3,9] + X[4,9] <= 1
 noBackToBack[3,10]: X[3,10] + X[4,10] <= 1
 noBackToBack[3,11]: X[3,11] + X[4,11] <= 1
 noBackToBack[3,12]: X[3,12] + X[4,12] <= 1
 noBackToBack[3,13]: X[3,13] + X[4,13] <= 1
 noBackToBack[4,1]: X[4,1] + X[5,1] <= 1
 noBackToBack[4,2]: X[4,2] + X[5,2] <= 1
 noBackToBack[4,3]: X[4,3] + X[5,3] <= 1
 noBackToBack[4,4]: X[4,4] + X[5,4] <= 1
 noBackToBack[4,5]: X[4,5] + X[5,5] <= 1
 noBackToBack[4,6]: X[4,6] + X[5,6] <= 1
 noBackToBack[4,7]: X[4,7] + X[5,7] <= 1
 noBackToBack[4,8]: X[4,8] + X[5,8] <= 1
 noBackToBack[4,9]: X[4,9] + X[5,9] <= 1
 noBackToBack[4,10]: X[4,10] + X[5,10] <= 1
 noBackToBack[4,11]: X[4,11] + X[5,11] <= 1
 noBackToBack[4,12]: X[4,12] + X[5,12] <= 1
 noBackToBack[4,13]: X[4,13] + X[5,13] <= 1
 noBackToBack[5,1]: X[5,1] + X[6,1] <= 1
 noBackToBack[5,2]: X[5,2] + X[6,2] <= 1
 noBackToBack[5,3]: X[5,3] + X[6,3] <= 1
 noBackToBack[5,4]: X[5,4] + X[6,4] <= 1
 noBackToBack[5,5]: X[5,5] + X[6,5] <= 1
 noBackToBack[5,6]: X[5,6] + X[6,6] <= 1
 noBackToBack[5,7]: X[5,7] + X[6,7] <= 1
 noBackToBack[5,8]: X[5,8] + X[6,8] <= 1
 noBackToBack[5,9]: X[5,9] + X[6,9] <= 1
 noBackToBack[5,10]: X[5,10] + X[6,10] <= 1
 noBackToBack[5,11]: X[5,11] + X[6,11] <= 1
 noBackToBack[5,12]: X[5,12] + X[6,12] <= 1
 noBackToBack[5,13]: X[5,13] + X[6,13] <= 1
 noBackToBack[6,1]: X[6,1] + X[7,1] <= 1
 noBackToBack[6,2]: X[6,2] + X[7,2] <= 1
 noBackToBack[6,3]: X[6,3] + X[7,3] <= 1
 noBackToBack[6,4]: X[6,4] + X[7,4] <= 1
 noBackToBack[6,5]: X[6,5] + X[7,5] <= 1
 noBackToBack[6,6]: X[6,6] + X[7,6] <= 1
 noBackToBack[6,7]: X[6,7] + X[7,7] <= 1
 noBackToBack[6,8]: X[6,8] + X[7,8] <= 1
 noBackToBack[6,9]: X[6,9] + X[7,9] <= 1
 noBackToBack[6,10]: X[6,10] + X[7,10] <= 1
 noBackToBack[6,11]: X[6,11] + X[7,11] <= 1
 noBackToBack[6,12]: X[6,12] + X[7,12] <= 1
 noBackToBack[6,13]: X[6,13] + X[7,13] <= 1
 noBackToBack[7,1]: X[7,1] + X[8,1] <= 1
 noBackToBack[7,2]: X[7,2] + X[8,2] <= 1
 noBackToBack[7,3]: X[7,3] + X[8,3] <= 1
 noBackToBack[7,4]: X[7,4] + X[8,4] <= 1
 noBackToBack[7,5]: X[7,5] + X[8,5] <= 1
 noBackToBack[7,6]: X[7,6] + X[8,6] <= 1
 noBackToBack[7,7]: X[7,7] + X[8,7] <= 1
 noBackToBack[7,8]: X[7,8] + X[8,8] <= 1
 noBackToBack[7,9]: X[7,9] + X[8,9] <= 1
 noBackToBack[7,10]: X[7,10] + X[8,10] <= 1
 noBackToBack[7,11]: X[7,11] + X[8,11] <= 1
 noBackToBack[7,12]: X[7,12] + X[8,12] <= 1
 noBackToBack[7,13]: X[7,13] + X[8,13] <= 1
 noBackToBack[8,1]: X[8,1] + X[9,1] <= 1
 noBackToBack[8,2]: X[8,2] + X[9,2] <= 1
 noBackToBack[8,3]: X[8,3] + X[9,3] <= 1
 noBackToBack[8,4]: X[8,4] + X[9,4] <= 1
 noBackToBack[8,5]: X[8,5] + X[9,5] <= 1
 noBackToBack[8,6]: X[8,6] + X[9,6] <= 1
 noBackToBack[8,7]: X[8,7] + X[9,7] <= 1
 noBackToBack[8,8]: X[8,8] + X[9,8] <= 1
 noBackToBack[8,9]: X[8,9] + X[9,9] <= 1
 noBackToBack[8,10]: X[8,10] + X[9,10] <= 1
 noBackToBack[8,11]: X[8,11] + X[9,11] <= 1
 noBackToBack[8,12]: X[8,12] + X[9,12] <= 1
 noBackToBack[8,13]: X[8,13] + X[9,13] <= 1
 noBackToBack[9,1]: X[9,1] + X[10,1] <= 1
 noBackToBack[9,2]: X[9,2] + X[10,2] <= 1
 noBackToBack[9,3]: X[9,3] + X[10,3] <= 1
 noBackToBack[9,4]: X[9,4] + X[10,4] <= 1
 noBackToBack[9,5]: X[9,5] + X[10,5] <= 1
 noBackToBack[9,6]: X[9,6] + X[10,6] <= 1
 noBackToBack[9,7]: X[9,7] + X[10,7] <= 1
 noBackToBack[9,8]: X[9,8] + X[10,8] <= 1
 noBackToBack[9,9]: X[9,9] + X[10,9] <= 1
 noBackToBack[9,10]: X[9,10] + X[10,10] <= 1
 noBackToBack[9,11]: X[9,11] + X[10,11] <= 1
 noBackToBack[9,12]: X[9,12] + X[10,12] <= 1
 noBackToBack[9,13]: X[9,13] + X[10,13] <= 1
 noBackToBack[10,1]: X[10,1] + X[11,1] <= 1
 noBackToBack[10,2]: X[10,2] + X[11,2] <= 1
 noBackToBack[10,3]: X[10,3] + X[11,3] <= 1
 noBackToBack[10,4]: X[10,4] + X[11,4] <= 1
 noBackToBack[10,5]: X[10,5] + X[11,5] <= 1
 noBackToBack[10,6]: X[10,6] + X[11,6] <= 1
 noBackToBack[10,7]: X[10,7] + X[11,7] <= 1
 noBackToBack[10,8]: X[10,8] + X[11,8] <= 1
 noBackToBack[10,9]: X[10,9] + X[11,9] <= 1
 noBackToBack[10,10]: X[10,10] + X[11,10] <= 1
 noBackToBack[10,11]: X[10,11] + X[11,11] <= 1
 noBackToBack[10,12]: X[10,12] + X[11,12] <= 1
 noBackToBack[10,13]: X[10,13] + X[11,13] <= 1
 noBackToBack[11,1]: X[11,1] + X[12,1] <= 1
 noBackToBack[11,2]: X[11,2] + X[12,2] <= 1
 noBackToBack[11,3]: X[11,3] + X[12,3] <= 1
 noBackToBack[11,4]: X[11,4] + X[12,4] <= 1
 noBackToBack[11,5]: X[11,5] + X[12,5] <= 1
 noBackToBack[11,6]: X[11,6] + X[12,6] <= 1
 noBackToBack[11,7]: X[11,7] + X[12,7] <= 1
 noBackToBack[11,8]: X[11,8] + X[12,8] <= 1
 noBackToBack[11,9]: X[11,9] + X[12,9] <= 1
 noBackToBack[11,10]: X[11,10] + X[12,10] <= 1
 noBackToBack[11,11]: X[11,11] + X[12,11] <= 1
 noBackToBack[11,12]: X[11,12] + X[12,12] <= 1
 noBackToBack[11,13]: X[11,13] + X[12,13] <= 1
 noBackToBack[12,1]: X[12,1] + X[13,1] <= 1
 noBackToBack[12,2]: X[12,2] + X[13,2] <= 1
 noBackToBack[12,3]: X[12,3] + X[13,3] <= 1
 noBackToBack[12,4]: X[12,4] + X[13,4] <= 1
 noBackToBack[12,5]: X[12,5] + X[13,5] <= 1
 noBackToBack[12,6]: X[12,6] + X[13,6] <= 1
 noBackToBack[12,7]: X[12,7] + X[13,7] <= 1
 noBackToBack[12,8]: X[12,8] + X[13,8] <= 1
 noBackToBack[12,9]: X[12,9] + X[13,9] <= 1
 noBackToBack[12,10]: X[12,10] + X[13,10] <= 1
 noBackToBack[12,11]: X[12,11] + X[13,11] <= 1
 noBackToBack[12,12]: X[12,12] + X[13,12] <= 1
 noBackToBack[12,13]: X[12,13] + X[13,13] <= 1
 noBackToBack[13,1]: X[13,1] + X[14,1] <= 1
 noBackToBack[13,2]: X[13,2] + X[14,2] <= 1
 noBackToBack[13,3]: X[13,3] + X[14,3] <= 1
 noBackToBack[13,4]: X[13,4] + X[14,4] <= 1
 noBackToBack[13,5]: X[13,5] + X[14,5] <= 1
 noBackToBack[13,6]: X[13,6] + X[14,6] <= 1
 noBackToBack[13,7]: X[13,7] + X[14,7] <= 1
 noBackToBack[13,8]: X[13,8] + X[14,8] <= 1
 noBackToBack[13,9]: X[13,9] + X[14,9] <= 1
 noBackToBack[13,10]: X[13,10] + X[14,10] <= 1
 noBackToBack[13,11]: X[13,11] + X[14,11] <= 1
 noBackToBack[13,12]: X[13,12] + X[14,12] <= 1
 noBackToBack[13,13]: X[13,13] + X[14,13] <= 1
 noBackToBack[14,1]: X[14,1] + X[15,1] <= 1
 noBackToBack[14,2]: X[14,2] + X[15,2] <= 1
 noBackToBack[14,3]: X[14,3] + X[15,3] <= 1
 noBackToBack[14,4]: X[14,4] + X[15,4] <= 1
 noBackToBack[14,5]: X[14,5] + X[15,5] <= 1
 noBackToBack[14,6]: X[14,6] + X[15,6] <= 1
 noBackToBack[14,7]: X[14,7] + X[15,7] <= 1
 noBackToBack[14,8]: X[14,8] + X[15,8] <= 1
 noBackToBack[14,9]: X[14,9] + X[15,9] <= 1
 noBackToBack[14,10]: X[14,10] + X[15,10] <= 1
 noBackToBack[14,11]: X[14,11] + X[15,11] <= 1
 noBackToBack[14,12]: X[14,12] + X[15,12] <= 1
 noBackToBack[14,13]: X[14,13] + X[15,13] <= 1
 noBackToBack[15,1]: X[15,1] + X[16,1] <= 1
 noBackToBack[15,2]: X[15,2] + X[16,2] <= 1
 noBackToBack[15,3]: X[15,3] + X[16,3] <= 1
 noBackToBack[15,4]: X[15,4] + X[16,4] <= 1
 noBackToBack[15,5]: X[15,5] + X[16,5] <= 1
 noBackToBack[15,6]: X[15,6] + X[16,6] <= 1
 noBackToBack[15,7]: X[15,7] + X[16,7] <= 1
 noBackToBack[15,8]: X[15,8] + X[16,8] <= 1
 noBackToBack[15,9]: X[15,9] + X[16,9] <= 1
 noBackToBack[15,10]: X[15,10] + X[16,10] <= 1
 noBackToBack[15,11]: X[15,11] + X[16,11] <= 1
 noBackToBack[15,12]: X[15,12] + X[16,12] <= 1
 noBackToBack[15,13]: X[15,13] + X[16,13] <= 1
 noBackToBack[16,1]: X[16,1] + X[17,1] <= 1
 noBackToBack[16,2]: X[16,2] + X[17,2] <= 1
 noBackToBack[16,3]: X[16,3] + X[17,3] <= 1
 noBackToBack[16,4]: X[16,4] + X[17,4] <= 1
 noBackToBack[16,5]: X[16,5] + X[17,5] <= 1
 noBackToBack[16,6]: X[16,6] + X[17,6] <= 1
 noBackToBack[16,7]: X[16,7] + X[17,7] <= 1
 noBackToBack[16,8]: X[16,8] + X[17,8] <= 1
 noBackToBack[16,9]: X[16,9] + X[17,9] <= 1
 noBackToBack[16,10]: X[16,10] + X[17,10] <= 1
 noBackToBack[16,11]: X[16,11] + X[17,11] <= 1
 noBackToBack[16,12]: X[16,12] + X[17,12] <= 1
 noBackToBack[16,13]: X[16,13] + X[17,13] <= 1
 noBackToBack[17,1]: X[17,1] + X[18,1] <= 1
 noBackToBack[17,2]: X[17,2] + X[18,2] <= 1
 noBackToBack[17,3]: X[17,3] + X[18,3] <= 1
 noBackToBack[17,4]: X[17,4] + X[18,4] <= 1
 noBackToBack[17,5]: X[17,5] + X[18,5] <= 1
 noBackToBack[17,6]: X[17,6] + X[18,6] <= 1
 noBackToBack[17,7]: X[17,7] + X[18,7] <= 1
 noBackToBack[17,8]: X[17,8] + X[18,8] <= 1
 noBackToBack[17,9]: X[17,9] + X[18,9] <= 1
 noBackToBack[17,10]: X[17,10] + X[18,10] <= 1
 noBackToBack[17,11]: X[17,11] + X[18,11] <= 1
 noBackToBack[17,12]: X[17,12] + X[18,12] <= 1
 noBackToBack[17,13]: X[17,13] + X[18,13] <= 1
 noBackToBack[18,1]: X[18,1] + X[19,1] <= 1
 noBackToBack[18,2]: X[18,2] + X[19,2] <= 1
 noBackToBack[18,3]: X[18,3] + X[19,3] <= 1
 noBackToBack[18,4]: X[18,4] + X[19,4] <= 1
 noBackToBack[18,5]: X[18,5] + X[19,5] <= 1
 noBackToBack[18,6]: X[18,6] + X[19,6] <= 1
 noBackToBack[18,7]: X[18,7] + X[19,7] <= 1
 noBackToBack[18,8]: X[18,8] + X[19,8] <= 1
 noBackToBack[18,9]: X[18,9] + X[19,9] <= 1
 noBackToBack[18,10]: X[18,10] + X[19,10] <= 1
 noBackToBack[18,11]: X[18,11] + X[19,11] <= 1
 noBackToBack[18,12]: X[18,12] + X[19,12] <= 1
 noBackToBack[18,13]: X[18,13] + X[19,13] <= 1
 noBackToBack[19,1]: X[19,1] + X[20,1] <= 1
 noBackToBack[19,2]: X[19,2] + X[20,2] <= 1
 noBackToBack[19,3]: X[19,3] + X[20,3] <= 1
 noBackToBack[19,4]: X[19,4] + X[20,4] <= 1
 noBackToBack[19,5]: X[19,5] + X[20,5] <= 1
 noBackToBack[19,6]: X[19,6] + X[20,6] <= 1
 noBackToBack[19,7]: X[19,7] + X[20,7] <= 1
 noBackToBack[19,8]: X[19,8] + X[20,8] <= 1
 noBackToBack[19,9]: X[19,9] + X[20,9] <= 1
 noBackToBack[19,10]: X[19,10] + X[20,10] <= 1
 noBackToBack[19,11]: X[19,11] + X[20,11] <= 1
 noBackToBack[19,12]: X[19,12] + X[20,12] <= 1
 noBackToBack[19,13]: X[19,13] + X[20,13] <= 1
 noBackToBack[20,1]: X[20,1] + X[21,1] <= 1
 noBackToBack[20,2]: X[20,2] + X[21,2] <= 1
 noBackToBack[20,3]: X[20,3] + X[21,3] <= 1
 noBackToBack[20,4]: X[20,4] + X[21,4] <= 1
 noBackToBack[20,5]: X[20,5] + X[21,5] <= 1
 noBackToBack[20,6]: X[20,6] + X[21,6] <= 1
 noBackToBack[20,7]: X[20,7] + X[21,7] <= 1
 noBackToBack[20,8]: X[20,8] + X[21,8] <= 1
 noBackToBack[20,9]: X[20,9] + X[21,9] <= 1
 noBackToBack[20,10]: X[20,10] + X[21,10] <= 1
 noBackToBack[20,11]: X[20,11] + X[21,11] <= 1
 noBackToBack[20,12]: X[20,12] + X[21,12] <= 1
 noBackToBack[20,13]: X[20,13] + X[21,13] <= 1
 noBackToBack[21,1]: X[21,1] + X[22,1] <= 1
 noBackToBack[21,2]: X[21,2] + X[22,2] <= 1
 noBackToBack[21,3]: X[21,3] + X[22,3] <= 1
 noBackToBack[21,4]: X[21,4] + X[22,4] <= 1
 noBackToBack[21,5]: X[21,5] + X[22,5] <= 1
 noBackToBack[21,6]: X[21,6] + X[22,6] <= 1
 noBackToBack[21,7]: X[21,7] + X[22,7] <= 1
 noBackToBack[21,8]: X[21,8] + X[22,8] <= 1
 noBackToBack[21,9]: X[21,9] + X[22,9] <= 1
 noBackToBack[21,10]: X[21,10] + X[22,10] <= 1
 noBackToBack[21,11]: X[21,11] + X[22,11] <= 1
 noBackToBack[21,12]: X[21,12] + X[22,12] <= 1
 noBackToBack[21,13]: X[21,13] + X[22,13] <= 1
 noBackToBack[22,1]: X[22,1] + X[23,1] <= 1
 noBackToBack[22,2]: X[22,2] + X[23,2] <= 1
 noBackToBack[22,3]: X[22,3] + X[23,3] <= 1
 noBackToBack[22,4]: X[22,4] + X[23,4] <= 1
 noBackToBack[22,5]: X[22,5] + X[23,5] <= 1
 noBackToBack[22,6]: X[22,6] + X[23,6] <= 1
 noBackToBack[22,7]: X[22,7] + X[23,7] <= 1
 noBackToBack[22,8]: X[22,8] + X[23,8] <= 1
 noBackToBack[22,9]: X[22,9] + X[23,9] <= 1
 noBackToBack[22,10]: X[22,10] + X[23,10] <= 1
 noBackToBack[22,11]: X[22,11] + X[23,11] <= 1
 noBackToBack[22,12]: X[22,12] + X[23,12] <= 1
 noBackToBack[22,13]: X[22,13] + X[23,13] <= 1
 noBackToBack[23,1]: X[23,1] + X[24,1] <= 1
 noBackToBack[23,2]: X[23,2] + X[24,2] <= 1
 noBackToBack[23,3]: X[23,3] + X[24,3] <= 1
 noBackToBack[23,4]: X[23,4] + X[24,4] <= 1
 noBackToBack[23,5]: X[23,5] + X[24,5] <= 1
 noBackToBack[23,6]: X[23,6] + X[24,6] <= 1
 noBackToBack[23,7]: X[23,7] + X[24,7] <= 1
 noBackToBack[23,8]: X[23,8] + X[24,8] <= 1
 noBackToBack[23,9]: X[23,9] + X[24,9] <= 1
 noBackToBack[23,10]: X[23,10] + X[24,10] <= 1
 noBackToBack[23,11]: X[23,11] + X[24,11] <= 1
 noBackToBack[23,12]: X[23,12] + X[24,12] <= 1
 noBackToBack[23,13]: X[23,13] + X[24,13] <= 1
 noBackToBack[24,1]: X[24,1] + X[25,1] <= 1
 noBackToBack[24,2]: X[24,2] + X[25,2] <= 1
 noBackToBack[24,3]: X[24,3] + X[25,3] <= 1
 noBackToBack[24,4]: X[24,4] + X[25,4] <= 1
 noBackToBack[24,5]: X[24,5] + X[25,5] <= 1
 noBackToBack[24,6]: X[24,6] + X[25,6] <= 1
 noBackToBack[24,7]: X[24,7] + X[25,7] <= 1
 noBackToBack[24,8]: X[24,8] + X[25,8] <= 1
 noBackToBack[24,9]: X[24,9] + X[25,9] <= 1
 noBackToBack[24,10]: X[24,10] + X[25,10] <= 1
 noBackToBack[24,11]: X[24,11] + X[25,11] <= 1
 noBackToBack[24,12]: X[24,12] + X[25,12] <= 1
 noBackToBack[24,13]: X[24,13] + X[25,13] <= 1
 noBackToBack[25,1]: X[25,1] + X[26,1] <= 1
 noBackToBack[25,2]: X[25,2] + X[26,2] <= 1
 noBackToBack[25,3]: X[25,3] + X[26,3] <= 1
 noBackToBack[25,4]: X[25,4] + X[26,4] <= 1
 noBackToBack[25,5]: X[25,5] + X[26,5] <= 1
 noBackToBack[25,6]: X[25,6] + X[26,6] <= 1
 noBackToBack[25,7]: X[25,7] + X[26,7] <= 1
 noBackToBack[25,8]: X[25,8] + X[26,8] <= 1
 noBackToBack[25,9]: X[25,9] + X[26,9] <= 1
 noBackToBack[25,10]: X[25,10] + X[26,10] <= 1
 noBackToBack[25,11]: X[25,11] + X[26,11] <= 1
 noBackToBack[25,12]: X[25,12] + X[26,12] <= 1
 noBackToBack[25,13]: X[25,13] + X[26,13] <= 1
 noBackToBack[26,1]: X[26,1] + X[27,1] <= 1
 noBackToBack[26,2]: X[26,2] + X[27,2] <= 1
 noBackToBack[26,3]: X[26,3] + X[27,3] <= 1
 noBackToBack[26,4]: X[26,4] + X[27,4] <= 1
 noBackToBack[26,5]: X[26,5] + X[27,5] <= 1
 noBackToBack[26,6]: X[26,6] + X[27,6] <= 1
 noBackToBack[26,7]: X[26,7] + X[27,7] <= 1
 noBackToBack[26,8]: X[26,8] + X[27,8] <= 1
 noBackToBack[26,9]: X[26,9] + X[27,9] <= 1
 noBackToBack[26,10]: X[26,10] + X[27,10] <= 1
 noBackToBack[26,11]: X[26,11] + X[27,11] <= 1
 noBackToBack[26,12]: X[26,12] + X[27,12] <= 1
 noBackToBack[26,13]: X[26,13] + X[27,13] <= 1
 noBackToBack[27,1]: X[27,1] + X[28,1] <= 1
 noBackToBack[27,2]: X[27,2] + X[28,2] <= 1
 noBackToBack[27,3]: X[27,3] + X[28,3] <= 1
 noBackToBack[27,4]: X[27,4] + X[28,4] <= 1
 noBackToBack[27,5]: X[27,5] + X[28,5] <= 1
 noBackToBack[27,6]: X[27,6] + X[28,6] <= 1
 noBackToBack[27,7]: X[27,7] + X[28,7] <= 1
 noBackToBack[27,8]: X[27,8] + X[28,8] <= 1
 noBackToBack[27,9]: X[27,9] + X[28,9] <= 1
 noBackToBack[27,10]: X[27,10] + X[28,10] <= 1
 noBackToBack[27,11]: X[27,11] + X[28,11] <= 1
 noBackToBack[27,12]: X[27,12] + X[28,12] <= 1
 noBackToBack[27,13]: X[27,13] + X[28,13] <= 1
 noBackToBack[28,1]: X[28,1] + X[29,1] <= 1
 noBackToBack[28,2]: X[28,2] + X[29,2] <= 1
 noBackToBack[28,3]: X[28,3] + X[29,3] <= 1
 noBackToBack[28,4]: X[28,4] + X[29,4] <= 1
 noBackToBack[28,5]: X[28,5] + X[29,5] <= 1
 noBackToBack[28,6]: X[28,6] + X[29,6] <= 1
 noBackToBack[28,7]: X[28,7] + X[29,7] <= 1
 noBackToBack[28,8]: X[28,8] + X[29,8] <= 1
 noBackToBack[28,9]: X[28,9] + X[29,9] <= 1
 noBackToBack[28,10]: X[28,10] + X[29,10] <= 1
 noBackToBack[28,11]: X[28,11] + X[29,11] <= 1
 noBackToBack[28,12]: X[28,12] + X[29,12] <= 1
 noBackToBack[28,13]: X[28,13] + X[29,13] <= 1
 noBackToBack[29,1]: X[29,1] + X[30,1] <= 1
 noBackToBack[29,2]: X[29,2] + X[30,2] <= 1
 noBackToBack[29,3]: X[29,3] + X[30,3] <= 1
 noBackToBack[29,4]: X[29,4] + X[30,4] <= 1
 noBackToBack[29,5]: X[29,5] + X[30,5] <= 1
 noBackToBack[29,6]: X[29,6] + X[30,6] <= 1
 noBackToBack[29,7]: X[29,7] + X[30,7] <= 1
 noBackToBack[29,8]: X[29,8] + X[30,8] <= 1
 noBackToBack[29,9]: X[29,9] + X[30,9] <= 1
 noBackToBack[29,10]: X[29,10] + X[30,10] <= 1
 noBackToBack[29,11]: X[29,11] + X[30,11] <= 1
 noBackToBack[29,12]: X[29,12] + X[30,12] <= 1
 noBackToBack[29,13]: X[29,13] + X[30,13] <= 1
 noBackToBack[30,1]: X[30,1] + X[31,1] <= 1
 noBackToBack[30,2]: X[30,2] + X[31,2] <= 1
 noBackToBack[30,3]: X[30,3] + X[31,3] <= 1
 noBackToBack[30,4]: X[30,4] + X[31,4] <= 1
 noBackToBack[30,5]: X[30,5] + X[31,5] <= 1
 noBackToBack[30,6]: X[30,6] + X[31,6] <= 1
 noBackToBack[30,7]: X[30,7] + X[31,7] <= 1
 noBackToBack[30,8]: X[30,8] + X[31,8] <= 1
 noBackToBack[30,9]: X[30,9] + X[31,9] <= 1
 noBackToBack[30,10]: X[30,10] + X[31,10] <= 1
 noBackToBack[30,11]: X[30,11] + X[31,11] <= 1
 noBackToBack[30,12]: X[30,12] + X[31,12] <= 1
 noBackToBack[30,13]: X[30,13] + X[31,13] <= 1
 noBackToBack[31,1]: X[31,1] + X[32,1] <= 1
 noBackToBack[31,2]: X[31,2] + X[32,2] <= 1
 noBackToBack[31,3]: X[31,3] + X[32,3] <= 1
 noBackToBack[31,4]: X[31,4] + X[32,4] <= 1
 noBackToBack[31,5]: X[31,5] + X[32,5] <= 1
 noBackToBack[31,6]: X[31,6] + X[32,6] <= 1
 noBackToBack[31,7]: X[31,7] + X[32,7] <= 1
 noBackToBack[31,8]: X[31,8] + X[32,8] <= 1
 noBackToBack[31,9]: X[31,9] + X[32,9] <= 1
 noBackToBack[31,10]: X[31,10] + X[32,10] <= 1
 noBackToBack[31,11]: X[31,11] + X[32,11] <= 1
 noBackToBack[31,12]: X[31,12] + X[32,12] <= 1
 noBackToBack[31,13]: X[31,13] + X[32,13] <= 1
 noBackToBack[32,1]: X[32,1] + X[33,1] <= 1
 noBackToBack[32,2]: X[32,2] + X[33,2] <= 1
 noBackToBack[32,3]: X[32,3] + X[33,3] <= 1
 noBackToBack[32,4]: X[32,4] + X[33,4] <= 1
 noBackToBack[32,5]: X[32,5] + X[33,5] <= 1
 noBackToBack[32,6]: X[32,6] + X[33,6] <= 1
 noBackToBack[32,7]: X[32,7] + X[33,7] <= 1
 noBackToBack[32,8]: X[32,8] + X[33,8] <= 1
 noBackToBack[32,9]: X[32,9] + X[33,9] <= 1
 noBackToBack[32,10]: X[32,10] + X[33,10] <= 1
 noBackToBack[32,11]: X[32,11] + X[33,11] <= 1
 noBackToBack[32,12]: X[32,12] + X[33,12] <= 1
 noBackToBack[32,13]: X[32,13] + X[33,13] <= 1
 noBackToBack[33,1]: X[33,1] + X[34,1] <= 1
 noBackToBack[33,2]: X[33,2] + X[34,2] <= 1
 noBackToBack[33,3]: X[33,3] + X[34,3] <= 1
 noBackToBack[33,4]: X[33,4] + X[34,4] <= 1
 noBackToBack[33,5]: X[33,5] + X[34,5] <= 1
 noBackToBack[33,6]: X[33,6] + X[34,6] <= 1
 noBackToBack[33,7]: X[33,7] + X[34,7] <= 1
 noBackToBack[33,8]: X[33,8] + X[34,8] <= 1
 noBackToBack[33,9]: X[33,9] + X[34,9] <= 1
 noBackToBack[33,10]: X[33,10] + X[34,10] <= 1
 noBackToBack[33,11]: X[33,11] + X[34,11] <= 1
 noBackToBack[33,12]: X[33,12] + X[34,12] <= 1
 noBackToBack[33,13]: X[33,13] + X[34,13] <= 1
 noBackToBack[34,1]: X[34,1] + X[35,1] <= 1
 noBackToBack[34,2]: X[34,2] + X[35,2] <= 1
 noBackToBack[34,3]: X[34,3] + X[35,3] <= 1
 noBackToBack[34,4]: X[34,4] + X[35,4] <= 1
 noBackToBack[34,5]: X[34,5] + X[35,5] <= 1
 noBackToBack[34,6]: X[34,6] + X[35,6] <= 1
 noBackToBack[34,7]: X[34,7] + X[35,7] <= 1
 noBackToBack[34,8]: X[34,8] + X[35,8] <= 1
 noBackToBack[34,9]: X[34,9] + X[35,9] <= 1
 noBackToBack[34,10]: X[34,10] + X[35,10] <= 1
 noBackToBack[34,11]: X[34,11] + X[35,11] <= 1
 noBackToBack[34,12]: X[34,12] + X[35,12] <= 1
 noBackToBack[34,13]: X[34,13] + X[35,13] <= 1
 noBackToBack[35,1]: X[35,1] + X[36,1] <= 1
 noBackToBack[35,2]: X[35,2] + X[36,2] <= 1
 noBackToBack[35,3]: X[35,3] + X[36,3] <= 1
 noBackToBack[35,4]: X[35,4] + X[36,4] <= 1
 noBackToBack[35,5]: X[35,5] + X[36,5] <= 1
 noBackToBack[35,6]: X[35,6] + X[36,6] <= 1
 noBackToBack[35,7]: X[35,7] + X[36,7] <= 1
 noBackToBack[35,8]: X[35,8] + X[36,8] <= 1
 noBackToBack[35,9]: X[35,9] + X[36,9] <= 1
 noBackToBack[35,10]: X[35,10] + X[36,10] <= 1
 noBackToBack[35,11]: X[35,11] + X[36,11] <= 1
 noBackToBack[35,12]: X[35,12] + X[36,12] <= 1
 noBackToBack[35,13]: X[35,13] + X[36,13] <= 1
 noBackToBack[36,1]: X[36,1] + X[37,1] <= 1
 noBackToBack[36,2]: X[36,2] + X[37,2] <= 1
 noBackToBack[36,3]: X[36,3] + X[37,3] <= 1
 noBackToBack[36,4]: X[36,4] + X[37,4] <= 1
 noBackToBack[36,5]: X[36,5] + X[37,5] <= 1
 noBackToBack[36,6]: X[36,6] + X[37,6] <= 1
 noBackToBack[36,7]: X[36,7] + X[37,7] <= 1
 noBackToBack[36,8]: X[36,8] + X[37,8] <= 1
 noBackToBack[36,9]: X[36,9] + X[37,9] <= 1
 noBackToBack[36,10]: X[36,10] + X[37,10] <= 1
 noBackToBack[36,11]: X[36,11] + X[37,11] <= 1
 noBackToBack[36,12]: X[36,12] + X[37,12] <= 1
 noBackToBack[36,13]: X[36,13] + X[37,13] <= 1
 noBackToBack[37,1]: X[37,1] + X[38,1] <= 1
 noBackToBack[37,2]: X[37,2] + X[38,2] <= 1
 noBackToBack[37,3]: X[37,3] + X[38,3] <= 1
 noBackToBack[37,4]: X[37,4] + X[38,4] <= 1
 noBackToBack[37,5]: X[37,5] + X[38,5] <= 1
 noBackToBack[37,6]: X[37,6] + X[38,6] <= 1
 noBackToBack[37,7]: X[37,7] + X[38,7] <= 1
 noBackToBack[37,8]: X[37,8] + X[38,8] <= 1
 noBackToBack[37,9]: X[37,9] + X[38,9] <= 1
 noBackToBack[37,10]: X[37,10] + X[38,10] <= 1
 noBackToBack[37,11]: X[37,11] + X[38,11] <= 1
 noBackToBack[37,12]: X[37,12] + X[38,12] <= 1
 noBackToBack[37,13]: X[37,13] + X[38,13] <= 1
 noBackToBack[38,1]: X[38,1] + X[39,1] <= 1
 noBackToBack[38,2]: X[38,2] + X[39,2] <= 1
 noBackToBack[38,3]: X[38,3] + X[39,3] <= 1
 noBackToBack[38,4]: X[38,4] + X[39,4] <= 1
 noBackToBack[38,5]: X[38,5] + X[39,5] <= 1
 noBackToBack[38,6]: X[38,6] + X[39,6] <= 1
 noBackToBack[38,7]: X[38,7] + X[39,7] <= 1
 noBackToBack[38,8]: X[38,8] + X[39,8] <= 1
 noBackToBack[38,9]: X[38,9] + X[39,9] <= 1
 noBackToBack[38,10]: X[38,10] + X[39,10] <= 1
 noBackToBack[38,11]: X[38,11] + X[39,11] <= 1
 noBackToBack[38,12]: X[38,12] + X[39,12] <= 1
 noBackToBack[38,13]: X[38,13] + X[39,13] <= 1
 noBackToBack[39,1]: X[39,1] + X[40,1] <= 1
 noBackToBack[39,2]: X[39,2] + X[40,2] <= 1
 noBackToBack[39,3]: X[39,3] + X[40,3] <= 1
 noBackToBack[39,4]: X[39,4] + X[40,4] <= 1
 noBackToBack[39,5]: X[39,5] + X[40,5] <= 1
 noBackToBack[39,6]: X[39,6] + X[40,6] <= 1
 noBackToBack[39,7]: X[39,7] + X[40,7] <= 1
 noBackToBack[39,8]: X[39,8] + X[40,8] <= 1
 noBackToBack[39,9]: X[39,9] + X[40,9] <= 1
 noBackToBack[39,10]: X[39,10] + X[40,10] <= 1
 noBackToBack[39,11]: X[39,11] + X[40,11] <= 1
 noBackToBack[39,12]: X[39,12] + X[40,12] <= 1
 noBackToBack[39,13]: X[39,13] + X[40,13] <= 1
 noBackToBack[40,1]: X[40,1] + X[41,1] <= 1
 noBackToBack[40,2]: X[40,2] + X[41,2] <= 1
 noBackToBack[40,3]: X[40,3] + X[41,3] <= 1
 noBackToBack[40,4]: X[40,4] + X[41,4] <= 1
 noBackToBack[40,5]: X[40,5] + X[41,5] <= 1
 noBackToBack[40,6]: X[40,6] + X[41,6] <= 1
 noBackToBack[40,7]: X[40,7] + X[41,7] <= 1
 noBackToBack[40,8]: X[40,8] + X[41,8] <= 1
 noBackToBack[40,9]: X[40,9] + X[41,9] <= 1
 noBackToBack[40,10]: X[40,10] + X[41,10] <= 1
 noBackToBack[40,11]: X[40,11] + X[41,11] <= 1
 noBackToBack[40,12]: X[40,12] + X[41,12] <= 1
 noBackToBack[40,13]: X[40,13] + X[41,13] <= 1
 noBackToBack[41,1]: X[41,1] + X[42,1] <= 1
 noBackToBack[41,2]: X[41,2] + X[42,2] <= 1
 noBackToBack[41,3]: X[41,3] + X[42,3] <= 1
 noBackToBack[41,4]: X[41,4] + X[42,4] <= 1
 noBackToBack[41,5]: X[41,5] + X[42,5] <= 1
 noBackToBack[41,6]: X[41,6] + X[42,6] <= 1
 noBackToBack[41,7]: X[41,7] + X[42,7] <= 1
 noBackToBack[41,8]: X[41,8] + X[42,8] <= 1
 noBackToBack[41,9]: X[41,9] + X[42,9] <= 1
 noBackToBack[41,10]: X[41,10] + X[42,10] <= 1
 noBackToBack[41,11]: X[41,11] + X[42,11] <= 1
 noBackToBack[41,12]: X[41,12] + X[42,12] <= 1
 noBackToBack[41,13]: X[41,13] + X[42,13] <= 1
 noConsecutiveNights[2,4,1]: X[2,1] + X[4,1] <= 1
 noConsecutiveNights[2,4,2]: X[2,2] + X[4,2] <= 1
 noConsecutiveNights[2,4,3]: X[2,3] + X[4,3] <= 1
 noConsecutiveNights[2,4,4]: X[2,4] + X[4,4] <= 1
 noConsecutiveNights[2,4,5]: X[2,5] + X[4,5] <= 1
 noConsecutiveNights[2,4,6]: X[2,6] + X[4,6] <= 1
 noConsecutiveNights[2,4,7]: X[2,7] + X[4,7] <= 1
 noConsecutiveNights[2,4,8]: X[2,8] + X[4,8] <= 1
 noConsecutiveNights[2,4,9]: X[2,9] + X[4,9] <= 1
 noConsecutiveNights[2,4,10]: X[2,10] + X[4,10] <= 1
 noConsecutiveNights[2,4,11]: X[2,11] + X[4,11] <= 1
 noConsecutiveNights[2,4,12]: X[2,12] + X[4,12] <= 1
 noConsecutiveNights[2,4,13]: X[2,13] + X[4,13] <= 1
 noConsecutiveNights[4,6,1]: X[4,1] + X[6,1] <= 1
 noConsecutiveNights[4,6,2]: X[4,2] + X[6,2] <= 1
 noConsecutiveNights[4,6,3]: X[4,3] + X[6,3] <= 1
 noConsecutiveNights[4,6,4]: X[4,4] + X[6,4] <= 1
 noConsecutiveNights[4,6,5]: X[4,5] + X[6,5] <= 1
 noConsecutiveNights[4,6,6]: X[4,6] + X[6,6] <= 1
 noConsecutiveNights[4,6,7]: X[4,7] + X[6,7] <= 1
 noConsecutiveNights[4,6,8]: X[4,8] + X[6,8] <= 1
 noConsecutiveNights[4,6,9]: X[4,9] + X[6,9] <= 1
 noConsecutiveNights[4,6,10]: X[4,10] + X[6,10] <= 1
 noConsecutiveNights[4,6,11]: X[4,11] + X[6,11] <= 1
 noConsecutiveNights[4,6,12]: X[4,12] + X[6,12] <= 1
 noConsecutiveNights[4,6,13]: X[4,13] + X[6,13] <= 1
 noConsecutiveNights[6,8,1]: X[6,1] + X[8,1] <= 1
 noConsecutiveNights[6,8,2]: X[6,2] + X[8,2] <= 1
 noConsecutiveNights[6,8,3]: X[6,3] + X[8,3] <= 1
 noConsecutiveNights[6,8,4]: X[6,4] + X[8,4] <= 1
 noConsecutiveNights[6,8,5]: X[6,5] + X[8,5] <= 1
 noConsecutiveNights[6,8,6]: X[6,6] + X[8,6] <= 1
 noConsecutiveNights[6,8,7]: X[6,7] + X[8,7] <= 1
 noConsecutiveNights[6,8,8]: X[6,8] + X[8,8] <= 1
 noConsecutiveNights[6,8,9]: X[6,9] + X[8,9] <= 1
 noConsecutiveNights[6,8,10]: X[6,10] + X[8,10] <= 1
 noConsecutiveNights[6,8,11]: X[6,11] + X[8,11] <= 1
 noConsecutiveNights[6,8,12]: X[6,12] + X[8,12] <= 1
 noConsecutiveNights[6,8,13]: X[6,13] + X[8,13] <= 1
 noConsecutiveNights[8,10,1]: X[8,1] + X[10,1] <= 1
 noConsecutiveNights[8,10,2]: X[8,2] + X[10,2] <= 1
 noConsecutiveNights[8,10,3]: X[8,3] + X[10,3] <= 1
 noConsecutiveNights[8,10,4]: X[8,4] + X[10,4] <= 1
 noConsecutiveNights[8,10,5]: X[8,5] + X[10,5] <= 1
 noConsecutiveNights[8,10,6]: X[8,6] + X[10,6] <= 1
 noConsecutiveNights[8,10,7]: X[8,7] + X[10,7] <= 1
 noConsecutiveNights[8,10,8]: X[8,8] + X[10,8] <= 1
 noConsecutiveNights[8,10,9]: X[8,9] + X[10,9] <= 1
 noConsecutiveNights[8,10,10]: X[8,10] + X[10,10] <= 1
 noConsecutiveNights[8,10,11]: X[8,11] + X[10,11] <= 1
 noConsecutiveNights[8,10,12]: X[8,12] + X[10,12] <= 1
 noConsecutiveNights[8,10,13]: X[8,13] + X[10,13] <= 1
 noConsecutiveNights[10,12,1]: X[10,1] + X[12,1] <= 1
 noConsecutiveNights[10,12,2]: X[10,2] + X[12,2] <= 1
 noConsecutiveNights[10,12,3]: X[10,3] + X[12,3] <= 1
 noConsecutiveNights[10,12,4]: X[10,4] + X[12,4] <= 1
 noConsecutiveNights[10,12,5]: X[10,5] + X[12,5] <= 1
 noConsecutiveNights[10,12,6]: X[10,6] + X[12,6] <= 1
 noConsecutiveNights[10,12,7]: X[10,7] + X[12,7] <= 1
 noConsecutiveNights[10,12,8]: X[10,8] + X[12,8] <= 1
 noConsecutiveNights[10,12,9]: X[10,9] + X[12,9] <= 1
 noConsecutiveNights[10,12,10]: X[10,10] + X[12,10] <= 1
 noConsecutiveNights[10,12,11]: X[10,11] + X[12,11] <= 1
 noConsecutiveNights[10,12,12]: X[10,12] + X[12,12] <= 1
 noConsecutiveNights[10,12,13]: X[10,13] + X[12,13] <= 1
 noConsecutiveNights[12,14,1]: X[12,1] + X[14,1] <= 1
 noConsecutiveNights[12,14,2]: X[12,2] + X[14,2] <= 1
 noConsecutiveNights[12,14,3]: X[12,3] + X[14,3] <= 1
 noConsecutiveNights[12,14,4]: X[12,4] + X[14,4] <= 1
 noConsecutiveNights[12,14,5]: X[12,5] + X[14,5] <= 1
 noConsecutiveNights[12,14,6]: X[12,6] + X[14,6] <= 1
 noConsecutiveNights[12,14,7]: X[12,7] + X[14,7] <= 1
 noConsecutiveNights[12,14,8]: X[12,8] + X[14,8] <= 1
 noConsecutiveNights[12,14,9]: X[12,9] + X[14,9] <= 1
 noConsecutiveNights[12,14,10]: X[12,10] + X[14,10] <= 1
 noConsecutiveNights[12,14,11]: X[12,11] + X[14,11] <= 1
 noConsecutiveNights[12,14,12]: X[12,12] + X[14,12] <= 1
 noConsecutiveNights[12,14,13]: X[12,13] + X[14,13] <= 1
 noConsecutiveNights[16,18,1]: X[16,1] + X[18,1] <= 1
 noConsecutiveNights[16,18,2]: X[16,2] + X[18,2] <= 1
 noConsecutiveNights[16,18,3]: X[16,3] + X[18,3] <= 1
 noConsecutiveNights[16,18,4]: X[16,4] + X[18,4] <= 1
 noConsecutiveNights[16,18,5]: X[16,5] + X[18,5] <= 1
 noConsecutiveNights[16,18,6]: X[16,6] + X[18,6] <= 1
 noConsecutiveNights[16,18,7]: X[16,7] + X[18,7] <= 1
 noConsecutiveNights[16,18,8]: X[16,8] + X[18,8] <= 1
 noConsecutiveNights[16,18,9]: X[16,9] + X[18,9] <= 1
 noConsecutiveNights[16,18,10]: X[16,10] + X[18,10] <= 1
 noConsecutiveNights[16,18,11]: X[16,11] + X[18,11] <= 1
 noConsecutiveNights[16,18,12]: X[16,12] + X[18,12] <= 1
 noConsecutiveNights[16,18,13]: X[16,13] + X[18,13] <= 1
 noConsecutiveNights[18,20,1]: X[18,1] + X[20,1] <= 1
 noConsecutiveNights[18,20,2]: X[18,2] + X[20,2] <= 1
 noConsecutiveNights[18,20,3]: X[18,3] + X[20,3] <= 1
 noConsecutiveNights[18,20,4]: X[18,4] + X[20,4] <= 1
 noConsecutiveNights[18,20,5]: X[18,5] + X[20,5] <= 1
 noConsecutiveNights[18,20,6]: X[18,6] + X[20,6] <= 1
 noConsecutiveNights[18,20,7]: X[18,7] + X[20,7] <= 1
 noConsecutiveNights[18,20,8]: X[18,8] + X[20,8] <= 1
 noConsecutiveNights[18,20,9]: X[18,9] + X[20,9] <= 1
 noConsecutiveNights[18,20,10]: X[18,10] + X[20,10] <= 1
 noConsecutiveNights[18,20,11]: X[18,11] + X[20,11] <= 1
 noConsecutiveNights[18,20,12]: X[18,12] + X[20,12] <= 1
 noConsecutiveNights[18,20,13]: X[18,13] + X[20,13] <= 1
 noConsecutiveNights[20,22,1]: X[20,1] + X[22,1] <= 1
 noConsecutiveNights[20,22,2]: X[20,2] + X[22,2] <= 1
 noConsecutiveNights[20,22,3]: X[20,3] + X[22,3] <= 1
 noConsecutiveNights[20,22,4]: X[20,4] + X[22,4] <= 1
 noConsecutiveNights[20,22,5]: X[20,5] + X[22,5] <= 1
 noConsecutiveNights[20,22,6]: X[20,6] + X[22,6] <= 1
 noConsecutiveNights[20,22,7]: X[20,7] + X[22,7] <= 1
 noConsecutiveNights[20,22,8]: X[20,8] + X[22,8] <= 1
 noConsecutiveNights[20,22,9]: X[20,9] + X[22,9] <= 1
 noConsecutiveNights[20,22,10]: X[20,10] + X[22,10] <= 1
 noConsecutiveNights[20,22,11]: X[20,11] + X[22,11] <= 1
 noConsecutiveNights[20,22,12]: X[20,12] + X[22,12] <= 1
 noConsecutiveNights[20,22,13]: X[20,13] + X[22,13] <= 1
 noConsecutiveNights[22,24,1]: X[22,1] + X[24,1] <= 1
 noConsecutiveNights[22,24,2]: X[22,2] + X[24,2] <= 1
 noConsecutiveNights[22,24,3]: X[22,3] + X[24,3] <= 1
 noConsecutiveNights[22,24,4]: X[22,4] + X[24,4] <= 1
 noConsecutiveNights[22,24,5]: X[22,5] + X[24,5] <= 1
 noConsecutiveNights[22,24,6]: X[22,6] + X[24,6] <= 1
 noConsecutiveNights[22,24,7]: X[22,7] + X[24,7] <= 1
 noConsecutiveNights[22,24,8]: X[22,8] + X[24,8] <= 1
 noConsecutiveNights[22,24,9]: X[22,9] + X[24,9] <= 1
 noConsecutiveNights[22,24,10]: X[22,10] + X[24,10] <= 1
 noConsecutiveNights[22,24,11]: X[22,11] + X[24,11] <= 1
 noConsecutiveNights[22,24,12]: X[22,12] + X[24,12] <= 1
 noConsecutiveNights[22,24,13]: X[22,13] + X[24,13] <= 1
 noConsecutiveNights[24,26,1]: X[24,1] + X[26,1] <= 1
 noConsecutiveNights[24,26,2]: X[24,2] + X[26,2] <= 1
 noConsecutiveNights[24,26,3]: X[24,3] + X[26,3] <= 1
 noConsecutiveNights[24,26,4]: X[24,4] + X[26,4] <= 1
 noConsecutiveNights[24,26,5]: X[24,5] + X[26,5] <= 1
 noConsecutiveNights[24,26,6]: X[24,6] + X[26,6] <= 1
 noConsecutiveNights[24,26,7]: X[24,7] + X[26,7] <= 1
 noConsecutiveNights[24,26,8]: X[24,8] + X[26,8] <= 1
 noConsecutiveNights[24,26,9]: X[24,9] + X[26,9] <= 1
 noConsecutiveNights[24,26,10]: X[24,10] + X[26,10] <= 1
 noConsecutiveNights[24,26,11]: X[24,11] + X[26,11] <= 1
 noConsecutiveNights[24,26,12]: X[24,12] + X[26,12] <= 1
 noConsecutiveNights[24,26,13]: X[24,13] + X[26,13] <= 1
 noConsecutiveNights[26,28,1]: X[26,1] + X[28,1] <= 1
 noConsecutiveNights[26,28,2]: X[26,2] + X[28,2] <= 1
 noConsecutiveNights[26,28,3]: X[26,3] + X[28,3] <= 1
 noConsecutiveNights[26,28,4]: X[26,4] + X[28,4] <= 1
 noConsecutiveNights[26,28,5]: X[26,5] + X[28,5] <= 1
 noConsecutiveNights[26,28,6]: X[26,6] + X[28,6] <= 1
 noConsecutiveNights[26,28,7]: X[26,7] + X[28,7] <= 1
 noConsecutiveNights[26,28,8]: X[26,8] + X[28,8] <= 1
 noConsecutiveNights[26,28,9]: X[26,9] + X[28,9] <= 1
 noConsecutiveNights[26,28,10]: X[26,10] + X[28,10] <= 1
 noConsecutiveNights[26,28,11]: X[26,11] + X[28,11] <= 1
 noConsecutiveNights[26,28,12]: X[26,12] + X[28,12] <= 1
 noConsecutiveNights[26,28,13]: X[26,13] + X[28,13] <= 1
 noConsecutiveNights[30,32,1]: X[30,1] + X[32,1] <= 1
 noConsecutiveNights[30,32,2]: X[30,2] + X[32,2] <= 1
 noConsecutiveNights[30,32,3]: X[30,3] + X[32,3] <= 1
 noConsecutiveNights[30,32,4]: X[30,4] + X[32,4] <= 1
 noConsecutiveNights[30,32,5]: X[30,5] + X[32,5] <= 1
 noConsecutiveNights[30,32,6]: X[30,6] + X[32,6] <= 1
 noConsecutiveNights[30,32,7]: X[30,7] + X[32,7] <= 1
 noConsecutiveNights[30,32,8]: X[30,8] + X[32,8] <= 1
 noConsecutiveNights[30,32,9]: X[30,9] + X[32,9] <= 1
 noConsecutiveNights[30,32,10]: X[30,10] + X[32,10] <= 1
 noConsecutiveNights[30,32,11]: X[30,11] + X[32,11] <= 1
 noConsecutiveNights[30,32,12]: X[30,12] + X[32,12] <= 1
 noConsecutiveNights[30,32,13]: X[30,13] + X[32,13] <= 1
 noConsecutiveNights[32,34,1]: X[32,1] + X[34,1] <= 1
 noConsecutiveNights[32,34,2]: X[32,2] + X[34,2] <= 1
 noConsecutiveNights[32,34,3]: X[32,3] + X[34,3] <= 1
 noConsecutiveNights[32,34,4]: X[32,4] + X[34,4] <= 1
 noConsecutiveNights[32,34,5]: X[32,5] + X[34,5] <= 1
 noConsecutiveNights[32,34,6]: X[32,6] + X[34,6] <= 1
 noConsecutiveNights[32,34,7]: X[32,7] + X[34,7] <= 1
 noConsecutiveNights[32,34,8]: X[32,8] + X[34,8] <= 1
 noConsecutiveNights[32,34,9]: X[32,9] + X[34,9] <= 1
 noConsecutiveNights[32,34,10]: X[32,10] + X[34,10] <= 1
 noConsecutiveNights[32,34,11]: X[32,11] + X[34,11] <= 1
 noConsecutiveNights[32,34,12]: X[32,12] + X[34,12] <= 1
 noConsecutiveNights[32,34,13]: X[32,13] + X[34,13] <= 1
 noConsecutiveNights[34,36,1]: X[34,1] + X[36,1] <= 1
 noConsecutiveNights[34,36,2]: X[34,2] + X[36,2] <= 1
 noConsecutiveNights[34,36,3]: X[34,3] + X[36,3] <= 1
 noConsecutiveNights[34,36,4]: X[34,4] + X[36,4] <= 1
 noConsecutiveNights[34,36,5]: X[34,5] + X[36,5] <= 1
 noConsecutiveNights[34,36,6]: X[34,6] + X[36,6] <= 1
 noConsecutiveNights[34,36,7]: X[34,7] + X[36,7] <= 1
 noConsecutiveNights[34,36,8]: X[34,8] + X[36,8] <= 1
 noConsecutiveNights[34,36,9]: X[34,9] + X[36,9] <= 1
 noConsecutiveNights[34,36,10]: X[34,10] + X[36,10] <= 1
 noConsecutiveNights[34,36,11]: X[34,11] + X[36,11] <= 1
 noConsecutiveNights[34,36,12]: X[34,12] + X[36,12] <= 1
 noConsecutiveNights[34,36,13]: X[34,13] + X[36,13] <= 1
 noConsecutiveNights[36,38,1]: X[36,1] + X[38,1] <= 1
 noConsecutiveNights[36,38,2]: X[36,2] + X[38,2] <= 1
 noConsecutiveNights[36,38,3]: X[36,3] + X[38,3] <= 1
 noConsecutiveNights[36,38,4]: X[36,4] + X[38,4] <= 1
 noConsecutiveNights[36,38,5]: X[36,5] + X[38,5] <= 1
 noConsecutiveNights[36,38,6]: X[36,6] + X[38,6] <= 1
 noConsecutiveNights[36,38,7]: X[36,7] + X[38,7] <= 1
 noConsecutiveNights[36,38,8]: X[36,8] + X[38,8] <= 1
 noConsecutiveNights[36,38,9]: X[36,9] + X[38,9] <= 1
 noConsecutiveNights[36,38,10]: X[36,10] + X[38,10] <= 1
 noConsecutiveNights[36,38,11]: X[36,11] + X[38,11] <= 1
 noConsecutiveNights[36,38,12]: X[36,12] + X[38,12] <= 1
 noConsecutiveNights[36,38,13]: X[36,13] + X[38,13] <= 1
 noConsecutiveNights[38,40,1]: X[38,1] + X[40,1] <= 1
 noConsecutiveNights[38,40,2]: X[38,2] + X[40,2] <= 1
 noConsecutiveNights[38,40,3]: X[38,3] + X[40,3] <= 1
 noConsecutiveNights[38,40,4]: X[38,4] + X[40,4] <= 1
 noConsecutiveNights[38,40,5]: X[38,5] + X[40,5] <= 1
 noConsecutiveNights[38,40,6]: X[38,6] + X[40,6] <= 1
 noConsecutiveNights[38,40,7]: X[38,7] + X[40,7] <= 1
 noConsecutiveNights[38,40,8]: X[38,8] + X[40,8] <= 1
 noConsecutiveNights[38,40,9]: X[38,9] + X[40,9] <= 1
 noConsecutiveNights[38,40,10]: X[38,10] + X[40,10] <= 1
 noConsecutiveNights[38,40,11]: X[38,11] + X[40,11] <= 1
 noConsecutiveNights[38,40,12]: X[38,12] + X[40,12] <= 1
 noConsecutiveNights[38,40,13]: X[38,13] + X[40,13] <= 1
 noConsecutiveNights[40,42,1]: X[40,1] + X[42,1] <= 1
 noConsecutiveNights[40,42,2]: X[40,2] + X[42,2] <= 1
 noConsecutiveNights[40,42,3]: X[40,3] + X[42,3] <= 1
 noConsecutiveNights[40,42,4]: X[40,4] + X[42,4] <= 1
 noConsecutiveNights[40,42,5]: X[40,5] + X[42,5] <= 1
 noConsecutiveNights[40,42,6]: X[40,6] + X[42,6] <= 1
 noConsecutiveNights[40,42,7]: X[40,7] + X[42,7] <= 1
 noConsecutiveNights[40,42,8]: X[40,8] + X[42,8] <= 1
 noConsecutiveNights[40,42,9]: X[40,9] + X[42,9] <= 1
 noConsecutiveNights[40,42,10]: X[40,10] + X[42,10] <= 1
 noConsecutiveNights[40,42,11]: X[40,11] + X[42,11] <= 1
 noConsecutiveNights[40,42,12]: X[40,12] + X[42,12] <= 1
 noConsecutiveNights[40,42,13]: X[40,13] + X[42,13] <= 1
 noConsecutiveNights[14,16,1]: X[14,1] + X[16,1] <= 1
 noConsecutiveNights[14,16,2]: X[14,2] + X[16,2] <= 1
 noConsecutiveNights[14,16,3]: X[14,3] + X[16,3] <= 1
 noConsecutiveNights[14,16,4]: X[14,4] + X[16,4] <= 1
 noConsecutiveNights[14,16,5]: X[14,5] + X[16,5] <= 1
 noConsecutiveNights[14,16,6]: X[14,6] + X[16,6] <= 1
 noConsecutiveNights[14,16,7]: X[14,7] + X[16,7] <= 1
 noConsecutiveNights[14,16,8]: X[14,8] + X[16,8] <= 1
 noConsecutiveNights[14,16,9]: X[14,9] + X[16,9] <= 1
 noConsecutiveNights[14,16,10]: X[14,10] + X[16,10] <= 1
 noConsecutiveNights[14,16,11]: X[14,11] + X[16,11] <= 1
 noConsecutiveNights[14,16,12]: X[14,12] + X[16,12] <= 1
 noConsecutiveNights[14,16,13]: X[14,13] + X[16,13] <= 1
 noConsecutiveNights[28,30,1]: X[28,1] + X[30,1] <= 1
 noConsecutiveNights[28,30,2]: X[28,2] + X[30,2] <= 1
 noConsecutiveNights[28,30,3]: X[28,3] + X[30,3] <= 1
 noConsecutiveNights[28,30,4]: X[28,4] + X[30,4] <= 1
 noConsecutiveNights[28,30,5]: X[28,5] + X[30,5] <= 1
 noConsecutiveNights[28,30,6]: X[28,6] + X[30,6] <= 1
 noConsecutiveNights[28,30,7]: X[28,7] + X[30,7] <= 1
 noConsecutiveNights[28,30,8]: X[28,8] + X[30,8] <= 1
 noConsecutiveNights[28,30,9]: X[28,9] + X[30,9] <= 1
 noConsecutiveNights[28,30,10]: X[28,10] + X[30,10] <= 1
 noConsecutiveNights[28,30,11]: X[28,11] + X[30,11] <= 1
 noConsecutiveNights[28,30,12]: X[28,12] + X[30,12] <= 1
 noConsecutiveNights[28,30,13]: X[28,13] + X[30,13] <= 1
 weekendCap[1,1]: X[10,1] + X[11,1] + X[12,1] + X[13,1] + X[14,1] <= 2
 weekendCap[1,2]: X[10,2] + X[11,2] + X[12,2] + X[13,2] + X[14,2] <= 2
 weekendCap[1,3]: X[10,3] + X[11,3] + X[12,3] + X[13,3] + X[14,3] <= 2
 weekendCap[1,4]: X[10,4] + X[11,4] + X[12,4] + X[13,4] + X[14,4] <= 2
 weekendCap[1,5]: X[10,5] + X[11,5] + X[12,5] + X[13,5] + X[14,5] <= 2
 weekendCap[1,6]: X[10,6] + X[11,6] + X[12,6] + X[13,6] + X[14,6] <= 2
 weekendCap[1,7]: X[10,7] + X[11,7] + X[12,7] + X[13,7] + X[14,7] <= 2
 weekendCap[1,8]: X[10,8] + X[11,8] + X[12,8] + X[13,8] + X[14,8] <= 2
 weekendCap[1,9]: X[10,9] + X[11,9] + X[12,9] + X[13,9] + X[14,9] <= 2
 weekendCap[1,10]: X[10,10] + X[11,10] + X[12,10] + X[13,10] + X[14,10]
   <= 2
 weekendCap[1,11]: X[10,11] + X[11,11] + X[12,11] + X[13,11] + X[14,11]
   <= 2
 weekendCap[1,12]: X[10,12] + X[11,12] + X[12,12] + X[13,12] + X[14,12]
   <= 2
 weekendCap[1,13]: X[10,13] + X[11,13] + X[12,13] + X[13,13] + X[14,13]
   <= 2
 weekendCap[2,1]: X[24,1] + X[25,1] + X[26,1] + X[27,1] + X[28,1] <= 2
 weekendCap[2,2]: X[24,2] + X[25,2] + X[26,2] + X[27,2] + X[28,2] <= 2
 weekendCap[2,3]: X[24,3] + X[25,3] + X[26,3] + X[27,3] + X[28,3] <= 2
 weekendCap[2,4]: X[24,4] + X[25,4] + X[26,4] + X[27,4] + X[28,4] <= 2
 weekendCap[2,5]: X[24,5] + X[25,5] + X[26,5] + X[27,5] + X[28,5] <= 2
 weekendCap[2,6]: X[24,6] + X[25,6] + X[26,6] + X[27,6] + X[28,6] <= 2
 weekendCap[2,7]: X[24,7] + X[25,7] + X[26,7] + X[27,7] + X[28,7] <= 2
 weekendCap[2,8]: X[24,8] + X[25,8] + X[26,8] + X[27,8] + X[28,8] <= 2
 weekendCap[2,9]: X[24,9] + X[25,9] + X[26,9] + X[27,9] + X[28,9] <= 2
 weekendCap[2,10]: X[24,10] + X[25,10] + X[26,10] + X[27,10] + X[28,10]
   <= 2
 weekendCap[2,11]: X[24,11] + X[25,11] + X[26,11] + X[27,11] + X[28,11]
   <= 2
 weekendCap[2,12]: X[24,12] + X[25,12] + X[26,12] + X[27,12] + X[28,12]
   <= 2
 weekendCap[2,13]: X[24,13] + X[25,13] + X[26,13] + X[27,13] + X[28,13]
   <= 2
 weekendCap[3,1]: X[38,1] + X[39,1] + X[40,1] + X[41,1] + X[42,1] <= 2
 weekendCap[3,2]: X[38,2] + X[39,2] + X[40,2] + X[41,2] + X[42,2] <= 2
 weekendCap[3,3]: X[38,3] + X[39,3] + X[40,3] + X[41,3] + X[42,3] <= 2
 weekendCap[3,4]: X[38,4] + X[39,4] + X[40,4] + X[41,4] + X[42,4] <= 2
 weekendCap[3,5]: X[38,5] + X[39,5] + X[40,5] + X[41,5] + X[42,5] <= 2
 weekendCap[3,6]: X[38,6] + X[39,6] + X[40,6] + X[41,6] + X[42,6] <= 2
 weekendCap[3,7]: X[38,7] + X[39,7] + X[40,7] + X[41,7] + X[42,7] <= 2
 weekendCap[3,8]: X[38,8] + X[39,8] + X[40,8] + X[41,8] + X[42,8] <= 2
 weekendCap[3,9]: X[38,9] + X[39,9] + X[40,9] + X[41,9] + X[42,9] <= 2
 weekendCap[3,10]: X[38,10] + X[39,10] + X[40,10] + X[41,10] + X[42,10]
   <= 2
 weekendCap[3,11]: X[38,11] + X[39,11] + X[40,11] + X[41,11] + X[42,11]
   <= 2
 weekendCap[3,12]: X[38,12] + X[39,12] + X[40,12] + X[41,12] + X[42,12]
   <= 2
 weekendCap[3,13]: X[38,13] + X[39,13] + X[40,13] + X[41,13] + X[42,13]
   <= 2
 onePOW[1]: Y[1,1] + Y[1,2] + Y[1,3] + Y[1,4] + Y[1,5] + Y[1,6] + Y[1,7]
   + Y[1,8] + Y[1,9] + Y[1,10] + Y[1,11] + Y[1,12] + Y[1,13] = 1
 onePOW[2]: Y[2,1] + Y[2,2] + Y[2,3] + Y[2,4] + Y[2,5] + Y[2,6] + Y[2,7]
   + Y[2,8] + Y[2,9] + Y[2,10] + Y[2,11] + Y[2,12] + Y[2,13] = 1
 onePOW[3]: Y[3,1] + Y[3,2] + Y[3,3] + Y[3,4] + Y[3,5] + Y[3,6] + Y[3,7]
   + Y[3,8] + Y[3,9] + Y[3,10] + Y[3,11] + Y[3,12] + Y[3,13] = 1
 powShiftCstr[1,1,3]: X[3,1] - Y[1,1] >= 0
 powShiftCstr[1,1,7]: X[7,1] - Y[1,1] >= 0
 powShiftCstr[1,1,12]: X[12,1] - Y[1,1] >= 0
 powShiftCstr[1,2,3]: X[3,2] - Y[1,2] >= 0
 powShiftCstr[1,2,7]: X[7,2] - Y[1,2] >= 0
 powShiftCstr[1,2,12]: X[12,2] - Y[1,2] >= 0
 powShiftCstr[1,3,3]: X[3,3] - Y[1,3] >= 0
 powShiftCstr[1,3,7]: X[7,3] - Y[1,3] >= 0
 powShiftCstr[1,3,12]: X[12,3] - Y[1,3] >= 0
 powShiftCstr[1,4,3]: X[3,4] - Y[1,4] >= 0
 powShiftCstr[1,4,7]: X[7,4] - Y[1,4] >= 0
 powShiftCstr[1,4,12]: X[12,4] - Y[1,4] >= 0
 powShiftCstr[1,5,3]: X[3,5] - Y[1,5] >= 0
 powShiftCstr[1,5,7]: X[7,5] - Y[1,5] >= 0
 powShiftCstr[1,5,12]: X[12,5] - Y[1,5] >= 0
 powShiftCstr[1,6,3]: X[3,6] - Y[1,6] >= 0
 powShiftCstr[1,6,7]: X[7,6] - Y[1,6] >= 0
 powShiftCstr[1,6,12]: X[12,6] - Y[1,6] >= 0
 powShiftCstr[1,7,3]: X[3,7] - Y[1,7] >= 0
 powShiftCstr[1,7,7]: X[7,7] - Y[1,7] >= 0
 powShiftCstr[1,7,12]: X[12,7] - Y[1,7] >= 0
 powShiftCstr[1,8,3]: X[3,8] - Y[1,8] >= 0
 powShiftCstr[1,8,7]: X[7,8] - Y[1,8] >= 0
 powShiftCstr[1,8,12]: X[12,8] - Y[1,8] >= 0
 powShiftCstr[1,9,3]: X[3,9] - Y[1,9] >= 0
 powShiftCstr[1,9,7]: X[7,9] - Y[1,9] >= 0
 powShiftCstr[1,9,12]: X[12,9] - Y[1,9] >= 0
 powShiftCstr[1,10,3]: X[3,10] - Y[1,10] >= 0
 powShiftCstr[1,10,7]: X[7,10] - Y[1,10] >= 0
 powShiftCstr[1,10,12]: X[12,10] - Y[1,10] >= 0
 powShiftCstr[1,11,3]: X[3,11] - Y[1,11] >= 0
 powShiftCstr[1,11,7]: X[7,11] - Y[1,11] >= 0
 powShiftCstr[1,11,12]: X[12,11] - Y[1,11] >= 0
 powShiftCstr[1,12,3]: X[3,12] - Y[1,12] >= 0
 powShiftCstr[1,12,7]: X[7,12] - Y[1,12] >= 0
 powShiftCstr[1,12,12]: X[12,12] - Y[1,12] >= 0
 powShiftCstr[1,13,3]: X[3,13] - Y[1,13] >= 0
 powShiftCstr[1,13,7]: X[7,13] - Y[1,13] >= 0
 powShiftCstr[1,13,12]: X[12,13] - Y[1,13] >= 0
 powShiftCstr[2,1,17]: X[17,1] - Y[2,1] >= 0
 powShiftCstr[2,1,21]: X[21,1] - Y[2,1] >= 0
 powShiftCstr[2,1,26]: X[26,1] - Y[2,1] >= 0
 powShiftCstr[2,2,17]: X[17,2] - Y[2,2] >= 0
 powShiftCstr[2,2,21]: X[21,2] - Y[2,2] >= 0
 powShiftCstr[2,2,26]: X[26,2] - Y[2,2] >= 0
 powShiftCstr[2,3,17]: X[17,3] - Y[2,3] >= 0
 powShiftCstr[2,3,21]: X[21,3] - Y[2,3] >= 0
 powShiftCstr[2,3,26]: X[26,3] - Y[2,3] >= 0
 powShiftCstr[2,4,17]: X[17,4] - Y[2,4] >= 0
 powShiftCstr[2,4,21]: X[21,4] - Y[2,4] >= 0
 powShiftCstr[2,4,26]: X[26,4] - Y[2,4] >= 0
 powShiftCstr[2,5,17]: X[17,5] - Y[2,5] >= 0
 powShiftCstr[2,5,21]: X[21,5] - Y[2,5] >= 0
 powShiftCstr[2,5,26]: X[26,5] - Y[2,5] >= 0
 powShiftCstr[2,6,17]: X[17,6] - Y[2,6] >= 0
 powShiftCstr[2,6,21]: X[21,6] - Y[2,6] >= 0
 powShiftCstr[2,6,26]: X[26,6] - Y[2,6] >= 0
 powShiftCstr[2,7,17]: X[17,7] - Y[2,7] >= 0
 powShiftCstr[2,7,21]: X[21,7] - Y[2,7] >= 0
 powShiftCstr[2,7,26]: X[26,7] - Y[2,7] >= 0
 powShiftCstr[2,8,17]: X[17,8] - Y[2,8] >= 0
 powShiftCstr[2,8,21]: X[21,8] - Y[2,8] >= 0
 powShiftCstr[2,8,26]: X[26,8] - Y[2,8] >= 0
 powShiftCstr[2,9,17]: X[17,9] - Y[2,9] >= 0
 powShiftCstr[2,9,21]: X[21,9] - Y[2,9] >= 0
 powShiftCstr[2,9,26]: X[26,9] - Y[2,9] >= 0
 powShiftCstr[2,10,17]: X[17,10] - Y[2,10] >= 0
 powShiftCstr[2,10,21]: X[21,10] - Y[2,10] >= 0
 powShiftCstr[2,10,26]: X[26,10] - Y[2,10] >= 0
 powShiftCstr[2,11,17]: X[17,11] - Y[2,11] >= 0
 powShiftCstr[2,11,21]: X[21,11] - Y[2,11] >= 0
 powShiftCstr[2,11,26]: X[26,11] - Y[2,11] >= 0
 powShiftCstr[2,12,17]: X[17,12] - Y[2,12] >= 0
 powShiftCstr[2,12,21]: X[21,12] - Y[2,12] >= 0
 powShiftCstr[2,12,26]: X[26,12] - Y[2,12] >= 0
 powShiftCstr[2,13,17]: X[17,13] - Y[2,13] >= 0
 powShiftCstr[2,13,21]: X[21,13] - Y[2,13] >= 0
 powShiftCstr[2,13,26]: X[26,13] - Y[2,13] >= 0
 powShiftCstr[3,1,31]: X[31,1] - Y[3,1] >= 0
 powShiftCstr[3,1,35]: X[35,1] - Y[3,1] >= 0
 powShiftCstr[3,1,40]: X[40,1] - Y[3,1] >= 0
 powShiftCstr[3,2,31]: X[31,2] - Y[3,2] >= 0
 powShiftCstr[3,2,35]: X[35,2] - Y[3,2] >= 0
 powShiftCstr[3,2,40]: X[40,2] - Y[3,2] >= 0
 powShiftCstr[3,3,31]: X[31,3] - Y[3,3] >= 0
 powShiftCstr[3,3,35]: X[35,3] - Y[3,3] >= 0
 powShiftCstr[3,3,40]: X[40,3] - Y[3,3] >= 0
 powShiftCstr[3,4,31]: X[31,4] - Y[3,4] >= 0
 powShiftCstr[3,4,35]: X[35,4] - Y[3,4] >= 0
 powShiftCstr[3,4,40]: X[40,4] - Y[3,4] >= 0
 powShiftCstr[3,5,31]: X[31,5] - Y[3,5] >= 0
 powShiftCstr[3,5,35]: X[35,5] - Y[3,5] >= 0
 powShiftCstr[3,5,40]: X[40,5] - Y[3,5] >= 0
 powShiftCstr[3,6,31]: X[31,6] - Y[3,6] >= 0
 powShiftCstr[3,6,35]: X[35,6] - Y[3,6] >= 0
 powShiftCstr[3,6,40]: X[40,6] - Y[3,6] >= 0
 powShiftCstr[3,7,31]: X[31,7] - Y[3,7] >= 0
 powShiftCstr[3,7,35]: X[35,7] - Y[3,7] >= 0
 powShiftCstr[3,7,40]: X[40,7] - Y[3,7] >= 0
 powShiftCstr[3,8,31]: X[31,8] - Y[3,8] >= 0
 powShiftCstr[3,8,35]: X[35,8] - Y[3,8] >= 0
 powShiftCstr[3,8,40]: X[40,8] - Y[3,8] >= 0
 powShiftCstr[3,9,31]: X[31,9] - Y[3,9] >= 0
 powShiftCstr[3,9,35]: X[35,9] - Y[3,9] >= 0
 powShiftCstr[3,9,40]: X[40,9] - Y[3,9] >= 0
 powShiftCstr[3,10,31]: X[31,10] - Y[3,10] >= 0
 powShiftCstr[3,10,35]: X[35,10] - Y[3,10] >= 0
 powShiftCstr[3,10,40]: X[40,10] - Y[3,10] >= 0
 powShiftCstr[3,11,31]: X[31,11] - Y[3,11] >= 0
 powShiftCstr[3,11,35]: X[35,11] - Y[3,11] >= 0
 powShiftCstr[3,11,40]: X[40,11] - Y[3,11] >= 0
 powShiftCstr[3,12,31]: X[31,12] - Y[3,12] >= 0
 powShiftCstr[3,12,35]: X[35,12] - Y[3,12] >= 0
 powShiftCstr[3,12,40]: X[40,12] - Y[3,12] >= 0
 powShiftCstr[3,13,31]: X[31,13] - Y[3,13] >= 0
 powShiftCstr[3,13,35]: X[35,13] - Y[3,13] >= 0
 powShiftCstr[3,13,40]: X[40,13] - Y[3,13] >= 0
 onePOWperCycle[1]: Y[1,1] + Y[2,1] + Y[3,1] <= 1
 onePOWperCycle[2]: Y[1,2] + Y[2,2] + Y[3,2] <= 1
 onePOWperCycle[3]: Y[1,3] + Y[2,3] + Y[3,3] <= 1
 onePOWperCycle[4]: Y[1,4] + Y[2,4] + Y[3,4] <= 1
 onePOWperCycle[5]: Y[1,5] + Y[2,5] + Y[3,5] <= 1
 onePOWperCycle[6]: Y[1,6] + Y[2,6] + Y[3,6] <= 1
 onePOWperCycle[7]: Y[1,7] + Y[2,7] + Y[3,7] <= 1
 onePOWperCycle[8]: Y[1,8] + Y[2,8] + Y[3,8] <= 1
 onePOWperCycle[9]: Y[1,9] + Y[2,9] + Y[3,9] <= 1
 onePOWperCycle[10]: Y[1,10] + Y[2,10] + Y[3,10] <= 1
 onePOWperCycle[11]: Y[1,11] + Y[2,11] + Y[3,11] <= 1
 onePOWperCycle[12]: Y[1,12] + Y[2,12] + Y[3,12] <= 1
 onePOWperCycle[13]: Y[1,13] + Y[2,13] + Y[3,13] <= 1
 POWfulfilled[3]: Y[1,3] + Y[2,3] + Y[3,3] = 0
 POWfulfilled[5]: Y[1,5] + Y[2,5] + Y[3,5] = 0
 POWfulfilled[13]: Y[1,13] + Y[2,13] + Y[3,13] = 0
 weekendLower[1,1]: - X[10,1] - X[11,1] - X[12,1] - X[13,1] - X[14,1]
   + Z[1,1] <= 0
 weekendLower[1,2]: - X[10,2] - X[11,2] - X[12,2] - X[13,2] - X[14,2]
   + Z[1,2] <= 0
 weekendLower[1,3]: - X[10,3] - X[11,3] - X[12,3] - X[13,3] - X[14,3]
   + Z[1,3] <= 0
 weekendLower[1,4]: - X[10,4] - X[11,4] - X[12,4] - X[13,4] - X[14,4]
   + Z[1,4] <= 0
 weekendLower[1,5]: - X[10,5] - X[11,5] - X[12,5] - X[13,5] - X[14,5]
   + Z[1,5] <= 0
 weekendLower[1,6]: - X[10,6] - X[11,6] - X[12,6] - X[13,6] - X[14,6]
   + Z[1,6] <= 0
 weekendLower[1,7]: - X[10,7] - X[11,7] - X[12,7] - X[13,7] - X[14,7]
   + Z[1,7] <= 0
 weekendLower[1,8]: - X[10,8] - X[11,8] - X[12,8] - X[13,8] - X[14,8]
   + Z[1,8] <= 0
 weekendLower[1,9]: - X[10,9] - X[11,9] - X[12,9] - X[13,9] - X[14,9]
   + Z[1,9] <= 0
 weekendLower[1,10]: - X[10,10] - X[11,10] - X[12,10] - X[13,10] - X[14,10]
   + Z[1,10] <= 0
 weekendLower[1,11]: - X[10,11] - X[11,11] - X[12,11] - X[13,11] - X[14,11]
   + Z[1,11] <= 0
 weekendLower[1,12]: - X[10,12] - X[11,12] - X[12,12] - X[13,12] - X[14,12]
   + Z[1,12] <= 0
 weekendLower[1,13]: - X[10,13] - X[11,13] - X[12,13] - X[13,13] - X[14,13]
   + Z[1,13] <= 0
 weekendLower[2,1]: - X[24,1] - X[25,1] - X[26,1] - X[27,1] - X[28,1]
   + Z[2,1] <= 0
 weekendLower[2,2]: - X[24,2] - X[25,2] - X[26,2] - X[27,2] - X[28,2]
   + Z[2,2] <= 0
 weekendLower[2,3]: - X[24,3] - X[25,3] - X[26,3] - X[27,3] - X[28,3]
   + Z[2,3] <= 0
 weekendLower[2,4]: - X[24,4] - X[25,4] - X[26,4] - X[27,4] - X[28,4]
   + Z[2,4] <= 0
 weekendLower[2,5]: - X[24,5] - X[25,5] - X[26,5] - X[27,5] - X[28,5]
   + Z[2,5] <= 0
 weekendLower[2,6]: - X[24,6] - X[25,6] - X[26,6] - X[27,6] - X[28,6]
   + Z[2,6] <= 0
 weekendLower[2,7]: - X[24,7] - X[25,7] - X[26,7] - X[27,7] - X[28,7]
   + Z[2,7] <= 0
 weekendLower[2,8]: - X[24,8] - X[25,8] - X[26,8] - X[27,8] - X[28,8]
   + Z[2,8] <= 0
 weekendLower[2,9]: - X[24,9] - X[25,9] - X[26,9] - X[27,9] - X[28,9]
   + Z[2,9] <= 0
 weekendLower[2,10]: - X[24,10] - X[25,10] - X[26,10] - X[27,10] - X[28,10]
   + Z[2,10] <= 0
 weekendLower[2,11]: - X[24,11] - X[25,11] - X[26,11] - X[27,11] - X[28,11]
   + Z[2,11] <= 0
 weekendLower[2,12]: - X[24,12] - X[25,12] - X[26,12] - X[27,12] - X[28,12]
   + Z[2,12] <= 0
 weekendLower[2,13]: - X[24,13] - X[25,13] - X[26,13] - X[27,13] - X[28,13]
   + Z[2,13] <= 0
 weekendLower[3,1]: - X[38,1] - X[39,1] - X[40,1] - X[41,1] - X[42,1]
   + Z[3,1] <= 0
 weekendLower[3,2]: - X[38,2] - X[39,2] - X[40,2] - X[41,2] - X[42,2]
   + Z[3,2] <= 0
 weekendLower[3,3]: - X[38,3] - X[39,3] - X[40,3] - X[41,3] - X[42,3]
   + Z[3,3] <= 0
 weekendLower[3,4]: - X[38,4] - X[39,4] - X[40,4] - X[41,4] - X[42,4]
   + Z[3,4] <= 0
 weekendLower[3,5]: - X[38,5] - X[39,5] - X[40,5] - X[41,5] - X[42,5]
   + Z[3,5] <= 0
 weekendLower[3,6]: - X[38,6] - X[39,6] - X[40,6] - X[41,6] - X[42,6]
   + Z[3,6] <= 0
 weekendLower[3,7]: - X[38,7] - X[39,7] - X[40,7] - X[41,7] - X[42,7]
   + Z[3,7] <= 0
 weekendLower[3,8]: - X[38,8] - X[39,8] - X[40,8] - X[41,8] - X[42,8]
   + Z[3,8] <= 0
 weekendLower[3,9]: - X[38,9] - X[39,9] - X[40,9] - X[41,9] - X[42,9]
   + Z[3,9] <= 0
 weekendLower[3,10]: - X[38,10] - X[39,10] - X[40,10] - X[41,10] - X[42,10]
   + Z[3,10] <= 0
 weekendLower[3,11]: - X[38,11] - X[39,11] - X[40,11] - X[41,11] - X[42,11]
   + Z[3,11] <= 0
 weekendLower[3,12]: - X[38,12] - X[39,12] - X[40,12] - X[41,12] - X[42,12]
   + Z[3,12] <= 0
 weekendLower[3,13]: - X[38,13] - X[39,13] - X[40,13] - X[41,13] - X[42,13]
   + Z[3,13] <= 0
 weekendUpper[1,1]: X[10,1] + X[11,1] + X[12,1] + X[13,1] + X[14,1]
   - 5 Z[1,1] <= 0
 weekendUpper[1,2]: X[10,2] + X[11,2] + X[12,2] + X[13,2] + X[14,2]
   - 5 Z[1,2] <= 0
 weekendUpper[1,3]: X[10,3] + X[11,3] + X[12,3] + X[13,3] + X[14,3]
   - 5 Z[1,3] <= 0
 weekendUpper[1,4]: X[10,4] + X[11,4] + X[12,4] + X[13,4] + X[14,4]
   - 5 Z[1,4] <= 0
 weekendUpper[1,5]: X[10,5] + X[11,5] + X[12,5] + X[13,5] + X[14,5]
   - 5 Z[1,5] <= 0
 weekendUpper[1,6]: X[10,6] + X[11,6] + X[12,6] + X[13,6] + X[14,6]
   - 5 Z[1,6] <= 0
 weekendUpper[1,7]: X[10,7] + X[11,7] + X[12,7] + X[13,7] + X[14,7]
   - 5 Z[1,7] <= 0
 weekendUpper[1,8]: X[10,8] + X[11,8] + X[12,8] + X[13,8] + X[14,8]
   - 5 Z[1,8] <= 0
 weekendUpper[1,9]: X[10,9] + X[11,9] + X[12,9] + X[13,9] + X[14,9]
   - 5 Z[1,9] <= 0
 weekendUpper[1,10]: X[10,10] + X[11,10] + X[12,10] + X[13,10] + X[14,10]
   - 5 Z[1,10] <= 0
 weekendUpper[1,11]: X[10,11] + X[11,11] + X[12,11] + X[13,11] + X[14,11]
   - 5 Z[1,11] <= 0
 weekendUpper[1,12]: X[10,12] + X[11,12] + X[12,12] + X[13,12] + X[14,12]
   - 5 Z[1,12] <= 0
 weekendUpper[1,13]: X[10,13] + X[11,13] + X[12,13] + X[13,13] + X[14,13]
   - 5 Z[1,13] <= 0
 weekendUpper[2,1]: X[24,1] + X[25,1] + X[26,1] + X[27,1] + X[28,1]
   - 5 Z[2,1] <= 0
 weekendUpper[2,2]: X[24,2] + X[25,2] + X[26,2] + X[27,2] + X[28,2]
   - 5 Z[2,2] <= 0
 weekendUpper[2,3]: X[24,3] + X[25,3] + X[26,3] + X[27,3] + X[28,3]
   - 5 Z[2,3] <= 0
 weekendUpper[2,4]: X[24,4] + X[25,4] + X[26,4] + X[27,4] + X[28,4]
   - 5 Z[2,4] <= 0
 weekendUpper[2,5]: X[24,5] + X[25,5] + X[26,5] + X[27,5] + X[28,5]
   - 5 Z[2,5] <= 0
 weekendUpper[2,6]: X[24,6] + X[25,6] + X[26,6] + X[27,6] + X[28,6]
   - 5 Z[2,6] <= 0
 weekendUpper[2,7]: X[24,7] + X[25,7] + X[26,7] + X[27,7] + X[28,7]
   - 5 Z[2,7] <= 0
 weekendUpper[2,8]: X[24,8] + X[25,8] + X[26,8] + X[27,8] + X[28,8]
   - 5 Z[2,8] <= 0
 weekendUpper[2,9]: X[24,9] + X[25,9] + X[26,9] + X[27,9] + X[28,9]
   - 5 Z[2,9] <= 0
 weekendUpper[2,10]: X[24,10] + X[25,10] + X[26,10] + X[27,10] + X[28,10]
   - 5 Z[2,10] <= 0
 weekendUpper[2,11]: X[24,11] + X[25,11] + X[26,11] + X[27,11] + X[28,11]
   - 5 Z[2,11] <= 0
 weekendUpper[2,12]: X[24,12] + X[25,12] + X[26,12] + X[27,12] + X[28,12]
   - 5 Z[2,12] <= 0
 weekendUpper[2,13]: X[24,13] + X[25,13] + X[26,13] + X[27,13] + X[28,13]
   - 5 Z[2,13] <= 0
 weekendUpper[3,1]: X[38,1] + X[39,1] + X[40,1] + X[41,1] + X[42,1]
   - 5 Z[3,1] <= 0
 weekendUpper[3,2]: X[38,2] + X[39,2] + X[40,2] + X[41,2] + X[42,2]
   - 5 Z[3,2] <= 0
 weekendUpper[3,3]: X[38,3] + X[39,3] + X[40,3] + X[41,3] + X[42,3]
   - 5 Z[3,3] <= 0
 weekendUpper[3,4]: X[38,4] + X[39,4] + X[40,4] + X[41,4] + X[42,4]
   - 5 Z[3,4] <= 0
 weekendUpper[3,5]: X[38,5] + X[39,5] + X[40,5] + X[41,5] + X[42,5]
   - 5 Z[3,5] <= 0
 weekendUpper[3,6]: X[38,6] + X[39,6] + X[40,6] + X[41,6] + X[42,6]
   - 5 Z[3,6] <= 0
 weekendUpper[3,7]: X[38,7] + X[39,7] + X[40,7] + X[41,7] + X[42,7]
   - 5 Z[3,7] <= 0
 weekendUpper[3,8]: X[38,8] + X[39,8] + X[40,8] + X[41,8] + X[42,8]
   - 5 Z[3,8] <= 0
 weekendUpper[3,9]: X[38,9] + X[39,9] + X[40,9] + X[41,9] + X[42,9]
   - 5 Z[3,9] <= 0
 weekendUpper[3,10]: X[38,10] + X[39,10] + X[40,10] + X[41,10] + X[42,10]
   - 5 Z[3,10] <= 0
 weekendUpper[3,11]: X[38,11] + X[39,11] + X[40,11] + X[41,11] + X[42,11]
   - 5 Z[3,11] <= 0
 weekendUpper[3,12]: X[38,12] + X[39,12] + X[40,12] + X[41,12] + X[42,12]
   - 5 Z[3,12] <= 0
 weekendUpper[3,13]: X[38,13] + X[39,13] + X[40,13] + X[41,13] + X[42,13]
   - 5 Z[3,13] <= 0
 consecWeekendPref[1,1]: - Z[1,1] - Z[2,1] + CW[1,1] >= -1
 consecWeekendPref[1,2]: - Z[1,2] - Z[2,2] + CW[1,2] >= -1
 consecWeekendPref[1,3]: - Z[1,3] - Z[2,3] + CW[1,3] >= -1
 consecWeekendPref[1,4]: - Z[1,4] - Z[2,4] + CW[1,4] >= -1
 consecWeekendPref[1,5]: - Z[1,5] - Z[2,5] + CW[1,5] >= -1
 consecWeekendPref[1,6]: - Z[1,6] - Z[2,6] + CW[1,6] >= -1
 consecWeekendPref[1,7]: - Z[1,7] - Z[2,7] + CW[1,7] >= -1
 consecWeekendPref[1,8]: - Z[1,8] - Z[2,8] + CW[1,8] >= -1
 consecWeekendPref[1,9]: - Z[1,9] - Z[2,9] + CW[1,9] >= -1
 consecWeekendPref[1,10]: - Z[1,10] - Z[2,10] + CW[1,10] >= -1
 consecWeekendPref[1,11]: - Z[1,11] - Z[2,11] + CW[1,11] >= -1
 consecWeekendPref[1,12]: - Z[1,12] - Z[2,12] + CW[1,12] >= -1
 consecWeekendPref[1,13]: - Z[1,13] - Z[2,13] + CW[1,13] >= -1
 consecWeekendPref[2,1]: - Z[2,1] - Z[3,1] + CW[2,1] >= -1
 consecWeekendPref[2,2]: - Z[2,2] - Z[3,2] + CW[2,2] >= -1
 consecWeekendPref[2,3]: - Z[2,3] - Z[3,3] + CW[2,3] >= -1
 consecWeekendPref[2,4]: - Z[2,4] - Z[3,4] + CW[2,4] >= -1
 consecWeekendPref[2,5]: - Z[2,5] - Z[3,5] + CW[2,5] >= -1
 consecWeekendPref[2,6]: - Z[2,6] - Z[3,6] + CW[2,6] >= -1
 consecWeekendPref[2,7]: - Z[2,7] - Z[3,7] + CW[2,7] >= -1
 consecWeekendPref[2,8]: - Z[2,8] - Z[3,8] + CW[2,8] >= -1
 consecWeekendPref[2,9]: - Z[2,9] - Z[3,9] + CW[2,9] >= -1
 consecWeekendPref[2,10]: - Z[2,10] - Z[3,10] + CW[2,10] >= -1
 consecWeekendPref[2,11]: - Z[2,11] - Z[3,11] + CW[2,11] >= -1
 consecWeekendPref[2,12]: - Z[2,12] - Z[3,12] + CW[2,12] >= -1
 consecWeekendPref[2,13]: - Z[2,13] - Z[3,13] + CW[2,13] >= -1
 twoApartNights[2,6,1]: X[2,1] + X[6,1] - E2[1,2,6] <= 1
 twoApartNights[2,6,2]: X[2,2] + X[6,2] - E2[2,2,6] <= 1
 twoApartNights[2,6,3]: X[2,3] + X[6,3] - E2[3,2,6] <= 1
 twoApartNights[2,6,4]: X[2,4] + X[6,4] - E2[4,2,6] <= 1
 twoApartNights[2,6,5]: X[2,5] + X[6,5] - E2[5,2,6] <= 1
 twoApartNights[2,6,6]: X[2,6] + X[6,6] - E2[6,2,6] <= 1
 twoApartNights[2,6,7]: X[2,7] + X[6,7] - E2[7,2,6] <= 1
 twoApartNights[2,6,8]: X[2,8] + X[6,8] - E2[8,2,6] <= 1
 twoApartNights[2,6,9]: X[2,9] + X[6,9] - E2[9,2,6] <= 1
 twoApartNights[2,6,10]: X[2,10] + X[6,10] - E2[10,2,6] <= 1
 twoApartNights[2,6,11]: X[2,11] + X[6,11] - E2[11,2,6] <= 1
 twoApartNights[2,6,12]: X[2,12] + X[6,12] - E2[12,2,6] <= 1
 twoApartNights[2,6,13]: X[2,13] + X[6,13] - E2[13,2,6] <= 1
 twoApartNights[4,8,1]: X[4,1] + X[8,1] - E2[1,4,8] <= 1
 twoApartNights[4,8,2]: X[4,2] + X[8,2] - E2[2,4,8] <= 1
 twoApartNights[4,8,3]: X[4,3] + X[8,3] - E2[3,4,8] <= 1
 twoApartNights[4,8,4]: X[4,4] + X[8,4] - E2[4,4,8] <= 1
 twoApartNights[4,8,5]: X[4,5] + X[8,5] - E2[5,4,8] <= 1
 twoApartNights[4,8,6]: X[4,6] + X[8,6] - E2[6,4,8] <= 1
 twoApartNights[4,8,7]: X[4,7] + X[8,7] - E2[7,4,8] <= 1
 twoApartNights[4,8,8]: X[4,8] + X[8,8] - E2[8,4,8] <= 1
 twoApartNights[4,8,9]: X[4,9] + X[8,9] - E2[9,4,8] <= 1
 twoApartNights[4,8,10]: X[4,10] + X[8,10] - E2[10,4,8] <= 1
 twoApartNights[4,8,11]: X[4,11] + X[8,11] - E2[11,4,8] <= 1
 twoApartNights[4,8,12]: X[4,12] + X[8,12] - E2[12,4,8] <= 1
 twoApartNights[4,8,13]: X[4,13] + X[8,13] - E2[13,4,8] <= 1
 twoApartNights[6,10,1]: X[6,1] + X[10,1] - E2[1,6,10] <= 1
 twoApartNights[6,10,2]: X[6,2] + X[10,2] - E2[2,6,10] <= 1
 twoApartNights[6,10,3]: X[6,3] + X[10,3] - E2[3,6,10] <= 1
 twoApartNights[6,10,4]: X[6,4] + X[10,4] - E2[4,6,10] <= 1
 twoApartNights[6,10,5]: X[6,5] + X[10,5] - E2[5,6,10] <= 1
 twoApartNights[6,10,6]: X[6,6] + X[10,6] - E2[6,6,10] <= 1
 twoApartNights[6,10,7]: X[6,7] + X[10,7] - E2[7,6,10] <= 1
 twoApartNights[6,10,8]: X[6,8] + X[10,8] - E2[8,6,10] <= 1
 twoApartNights[6,10,9]: X[6,9] + X[10,9] - E2[9,6,10] <= 1
 twoApartNights[6,10,10]: X[6,10] + X[10,10] - E2[10,6,10] <= 1
 twoApartNights[6,10,11]: X[6,11] + X[10,11] - E2[11,6,10] <= 1
 twoApartNights[6,10,12]: X[6,12] + X[10,12] - E2[12,6,10] <= 1
 twoApartNights[6,10,13]: X[6,13] + X[10,13] - E2[13,6,10] <= 1
 twoApartNights[8,12,1]: X[8,1] + X[12,1] - E2[1,8,12] <= 1
 twoApartNights[8,12,2]: X[8,2] + X[12,2] - E2[2,8,12] <= 1
 twoApartNights[8,12,3]: X[8,3] + X[12,3] - E2[3,8,12] <= 1
 twoApartNights[8,12,4]: X[8,4] + X[12,4] - E2[4,8,12] <= 1
 twoApartNights[8,12,5]: X[8,5] + X[12,5] - E2[5,8,12] <= 1
 twoApartNights[8,12,6]: X[8,6] + X[12,6] - E2[6,8,12] <= 1
 twoApartNights[8,12,7]: X[8,7] + X[12,7] - E2[7,8,12] <= 1
 twoApartNights[8,12,8]: X[8,8] + X[12,8] - E2[8,8,12] <= 1
 twoApartNights[8,12,9]: X[8,9] + X[12,9] - E2[9,8,12] <= 1
 twoApartNights[8,12,10]: X[8,10] + X[12,10] - E2[10,8,12] <= 1
 twoApartNights[8,12,11]: X[8,11] + X[12,11] - E2[11,8,12] <= 1
 twoApartNights[8,12,12]: X[8,12] + X[12,12] - E2[12,8,12] <= 1
 twoApartNights[8,12,13]: X[8,13] + X[12,13] - E2[13,8,12] <= 1
 twoApartNights[10,14,1]: X[10,1] + X[14,1] - E2[1,10,14] <= 1
 twoApartNights[10,14,2]: X[10,2] + X[14,2] - E2[2,10,14] <= 1
 twoApartNights[10,14,3]: X[10,3] + X[14,3] - E2[3,10,14] <= 1
 twoApartNights[10,14,4]: X[10,4] + X[14,4] - E2[4,10,14] <= 1
 twoApartNights[10,14,5]: X[10,5] + X[14,5] - E2[5,10,14] <= 1
 twoApartNights[10,14,6]: X[10,6] + X[14,6] - E2[6,10,14] <= 1
 twoApartNights[10,14,7]: X[10,7] + X[14,7] - E2[7,10,14] <= 1
 twoApartNights[10,14,8]: X[10,8] + X[14,8] - E2[8,10,14] <= 1
 twoApartNights[10,14,9]: X[10,9] + X[14,9] - E2[9,10,14] <= 1
 twoApartNights[10,14,10]: X[10,10] + X[14,10] - E2[10,10,14] <= 1
 twoApartNights[10,14,11]: X[10,11] + X[14,11] - E2[11,10,14] <= 1
 twoApartNights[10,14,12]: X[10,12] + X[14,12] - E2[12,10,14] <= 1
 twoApartNights[10,14,13]: X[10,13] + X[14,13] - E2[13,10,14] <= 1
 twoApartNights[16,20,1]: X[16,1] + X[20,1] - E2[1,16,20] <= 1
 twoApartNights[16,20,2]: X[16,2] + X[20,2] - E2[2,16,20] <= 1
 twoApartNights[16,20,3]: X[16,3] + X[20,3] - E2[3,16,20] <= 1
 twoApartNights[16,20,4]: X[16,4] + X[20,4] - E2[4,16,20] <= 1
 twoApartNights[16,20,5]: X[16,5] + X[20,5] - E2[5,16,20] <= 1
 twoApartNights[16,20,6]: X[16,6] + X[20,6] - E2[6,16,20] <= 1
 twoApartNights[16,20,7]: X[16,7] + X[20,7] - E2[7,16,20] <= 1
 twoApartNights[16,20,8]: X[16,8] + X[20,8] - E2[8,16,20] <= 1
 twoApartNights[16,20,9]: X[16,9] + X[20,9] - E2[9,16,20] <= 1
 twoApartNights[16,20,10]: X[16,10] + X[20,10] - E2[10,16,20] <= 1
 twoApartNights[16,20,11]: X[16,11] + X[20,11] - E2[11,16,20] <= 1
 twoApartNights[16,20,12]: X[16,12] + X[20,12] - E2[12,16,20] <= 1
 twoApartNights[16,20,13]: X[16,13] + X[20,13] - E2[13,16,20] <= 1
 twoApartNights[18,22,1]: X[18,1] + X[22,1] - E2[1,18,22] <= 1
 twoApartNights[18,22,2]: X[18,2] + X[22,2] - E2[2,18,22] <= 1
 twoApartNights[18,22,3]: X[18,3] + X[22,3] - E2[3,18,22] <= 1
 twoApartNights[18,22,4]: X[18,4] + X[22,4] - E2[4,18,22] <= 1
 twoApartNights[18,22,5]: X[18,5] + X[22,5] - E2[5,18,22] <= 1
 twoApartNights[18,22,6]: X[18,6] + X[22,6] - E2[6,18,22] <= 1
 twoApartNights[18,22,7]: X[18,7] + X[22,7] - E2[7,18,22] <= 1
 twoApartNights[18,22,8]: X[18,8] + X[22,8] - E2[8,18,22] <= 1
 twoApartNights[18,22,9]: X[18,9] + X[22,9] - E2[9,18,22] <= 1
 twoApartNights[18,22,10]: X[18,10] + X[22,10] - E2[10,18,22] <= 1
 twoApartNights[18,22,11]: X[18,11] + X[22,11] - E2[11,18,22] <= 1
 twoApartNights[18,22,12]: X[18,12] + X[22,12] - E2[12,18,22] <= 1
 twoApartNights[18,22,13]: X[18,13] + X[22,13] - E2[13,18,22] <= 1
 twoApartNights[20,24,1]: X[20,1] + X[24,1] - E2[1,20,24] <= 1
 twoApartNights[20,24,2]: X[20,2] + X[24,2] - E2[2,20,24] <= 1
 twoApartNights[20,24,3]: X[20,3] + X[24,3] - E2[3,20,24] <= 1
 twoApartNights[20,24,4]: X[20,4] + X[24,4] - E2[4,20,24] <= 1
 twoApartNights[20,24,5]: X[20,5] + X[24,5] - E2[5,20,24] <= 1
 twoApartNights[20,24,6]: X[20,6] + X[24,6] - E2[6,20,24] <= 1
 twoApartNights[20,24,7]: X[20,7] + X[24,7] - E2[7,20,24] <= 1
 twoApartNights[20,24,8]: X[20,8] + X[24,8] - E2[8,20,24] <= 1
 twoApartNights[20,24,9]: X[20,9] + X[24,9] - E2[9,20,24] <= 1
 twoApartNights[20,24,10]: X[20,10] + X[24,10] - E2[10,20,24] <= 1
 twoApartNights[20,24,11]: X[20,11] + X[24,11] - E2[11,20,24] <= 1
 twoApartNights[20,24,12]: X[20,12] + X[24,12] - E2[12,20,24] <= 1
 twoApartNights[20,24,13]: X[20,13] + X[24,13] - E2[13,20,24] <= 1
 twoApartNights[22,26,1]: X[22,1] + X[26,1] - E2[1,22,26] <= 1
 twoApartNights[22,26,2]: X[22,2] + X[26,2] - E2[2,22,26] <= 1
 twoApartNights[22,26,3]: X[22,3] + X[26,3] - E2[3,22,26] <= 1
 twoApartNights[22,26,4]: X[22,4] + X[26,4] - E2[4,22,26] <= 1
 twoApartNights[22,26,5]: X[22,5] + X[26,5] - E2[5,22,26] <= 1
 twoApartNights[22,26,6]: X[22,6] + X[26,6] - E2[6,22,26] <= 1
 twoApartNights[22,26,7]: X[22,7] + X[26,7] - E2[7,22,26] <= 1
 twoApartNights[22,26,8]: X[22,8] + X[26,8] - E2[8,22,26] <= 1
 twoApartNights[22,26,9]: X[22,9] + X[26,9] - E2[9,22,26] <= 1
 twoApartNights[22,26,10]: X[22,10] + X[26,10] - E2[10,22,26] <= 1
 twoApartNights[22,26,11]: X[22,11] + X[26,11] - E2[11,22,26] <= 1
 twoApartNights[22,26,12]: X[22,12] + X[26,12] - E2[12,22,26] <= 1
 twoApartNights[22,26,13]: X[22,13] + X[26,13] - E2[13,22,26] <= 1
 twoApartNights[24,28,1]: X[24,1] + X[28,1] - E2[1,24,28] <= 1
 twoApartNights[24,28,2]: X[24,2] + X[28,2] - E2[2,24,28] <= 1
 twoApartNights[24,28,3]: X[24,3] + X[28,3] - E2[3,24,28] <= 1
 twoApartNights[24,28,4]: X[24,4] + X[28,4] - E2[4,24,28] <= 1
 twoApartNights[24,28,5]: X[24,5] + X[28,5] - E2[5,24,28] <= 1
 twoApartNights[24,28,6]: X[24,6] + X[28,6] - E2[6,24,28] <= 1
 twoApartNights[24,28,7]: X[24,7] + X[28,7] - E2[7,24,28] <= 1
 twoApartNights[24,28,8]: X[24,8] + X[28,8] - E2[8,24,28] <= 1
 twoApartNights[24,28,9]: X[24,9] + X[28,9] - E2[9,24,28] <= 1
 twoApartNights[24,28,10]: X[24,10] + X[28,10] - E2[10,24,28] <= 1
 twoApartNights[24,28,11]: X[24,11] + X[28,11] - E2[11,24,28] <= 1
 twoApartNights[24,28,12]: X[24,12] + X[28,12] - E2[12,24,28] <= 1
 twoApartNights[24,28,13]: X[24,13] + X[28,13] - E2[13,24,28] <= 1
 twoApartNights[30,34,1]: X[30,1] + X[34,1] - E2[1,30,34] <= 1
 twoApartNights[30,34,2]: X[30,2] + X[34,2] - E2[2,30,34] <= 1
 twoApartNights[30,34,3]: X[30,3] + X[34,3] - E2[3,30,34] <= 1
 twoApartNights[30,34,4]: X[30,4] + X[34,4] - E2[4,30,34] <= 1
 twoApartNights[30,34,5]: X[30,5] + X[34,5] - E2[5,30,34] <= 1
 twoApartNights[30,34,6]: X[30,6] + X[34,6] - E2[6,30,34] <= 1
 twoApartNights[30,34,7]: X[30,7] + X[34,7] - E2[7,30,34] <= 1
 twoApartNights[30,34,8]: X[30,8] + X[34,8] - E2[8,30,34] <= 1
 twoApartNights[30,34,9]: X[30,9] + X[34,9] - E2[9,30,34] <= 1
 twoApartNights[30,34,10]: X[30,10] + X[34,10] - E2[10,30,34] <= 1
 twoApartNights[30,34,11]: X[30,11] + X[34,11] - E2[11,30,34] <= 1
 twoApartNights[30,34,12]: X[30,12] + X[34,12] - E2[12,30,34] <= 1
 twoApartNights[30,34,13]: X[30,13] + X[34,13] - E2[13,30,34] <= 1
 twoApartNights[32,36,1]: X[32,1] + X[36,1] - E2[1,32,36] <= 1
 twoApartNights[32,36,2]: X[32,2] + X[36,2] - E2[2,32,36] <= 1
 twoApartNights[32,36,3]: X[32,3] + X[36,3] - E2[3,32,36] <= 1
 twoApartNights[32,36,4]: X[32,4] + X[36,4] - E2[4,32,36] <= 1
 twoApartNights[32,36,5]: X[32,5] + X[36,5] - E2[5,32,36] <= 1
 twoApartNights[32,36,6]: X[32,6] + X[36,6] - E2[6,32,36] <= 1
 twoApartNights[32,36,7]: X[32,7] + X[36,7] - E2[7,32,36] <= 1
 twoApartNights[32,36,8]: X[32,8] + X[36,8] - E2[8,32,36] <= 1
 twoApartNights[32,36,9]: X[32,9] + X[36,9] - E2[9,32,36] <= 1
 twoApartNights[32,36,10]: X[32,10] + X[36,10] - E2[10,32,36] <= 1
 twoApartNights[32,36,11]: X[32,11] + X[36,11] - E2[11,32,36] <= 1
 twoApartNights[32,36,12]: X[32,12] + X[36,12] - E2[12,32,36] <= 1
 twoApartNights[32,36,13]: X[32,13] + X[36,13] - E2[13,32,36] <= 1
 twoApartNights[34,38,1]: X[34,1] + X[38,1] - E2[1,34,38] <= 1
 twoApartNights[34,38,2]: X[34,2] + X[38,2] - E2[2,34,38] <= 1
 twoApartNights[34,38,3]: X[34,3] + X[38,3] - E2[3,34,38] <= 1
 twoApartNights[34,38,4]: X[34,4] + X[38,4] - E2[4,34,38] <= 1
 twoApartNights[34,38,5]: X[34,5] + X[38,5] - E2[5,34,38] <= 1
 twoApartNights[34,38,6]: X[34,6] + X[38,6] - E2[6,34,38] <= 1
 twoApartNights[34,38,7]: X[34,7] + X[38,7] - E2[7,34,38] <= 1
 twoApartNights[34,38,8]: X[34,8] + X[38,8] - E2[8,34,38] <= 1
 twoApartNights[34,38,9]: X[34,9] + X[38,9] - E2[9,34,38] <= 1
 twoApartNights[34,38,10]: X[34,10] + X[38,10] - E2[10,34,38] <= 1
 twoApartNights[34,38,11]: X[34,11] + X[38,11] - E2[11,34,38] <= 1
 twoApartNights[34,38,12]: X[34,12] + X[38,12] - E2[12,34,38] <= 1
 twoApartNights[34,38,13]: X[34,13] + X[38,13] - E2[13,34,38] <= 1
 twoApartNights[36,40,1]: X[36,1] + X[40,1] - E2[1,36,40] <= 1
 twoApartNights[36,40,2]: X[36,2] + X[40,2] - E2[2,36,40] <= 1
 twoApartNights[36,40,3]: X[36,3] + X[40,3] - E2[3,36,40] <= 1
 twoApartNights[36,40,4]: X[36,4] + X[40,4] - E2[4,36,40] <= 1
 twoApartNights[36,40,5]: X[36,5] + X[40,5] - E2[5,36,40] <= 1
 twoApartNights[36,40,6]: X[36,6] + X[40,6] - E2[6,36,40] <= 1
 twoApartNights[36,40,7]: X[36,7] + X[40,7] - E2[7,36,40] <= 1
 twoApartNights[36,40,8]: X[36,8] + X[40,8] - E2[8,36,40] <= 1
 twoApartNights[36,40,9]: X[36,9] + X[40,9] - E2[9,36,40] <= 1
 twoApartNights[36,40,10]: X[36,10] + X[40,10] - E2[10,36,40] <= 1
 twoApartNights[36,40,11]: X[36,11] + X[40,11] - E2[11,36,40] <= 1
 twoApartNights[36,40,12]: X[36,12] + X[40,12] - E2[12,36,40] <= 1
 twoApartNights[36,40,13]: X[36,13] + X[40,13] - E2[13,36,40] <= 1
 twoApartNights[38,42,1]: X[38,1] + X[42,1] - E2[1,38,42] <= 1
 twoApartNights[38,42,2]: X[38,2] + X[42,2] - E2[2,38,42] <= 1
 twoApartNights[38,42,3]: X[38,3] + X[42,3] - E2[3,38,42] <= 1
 twoApartNights[38,42,4]: X[38,4] + X[42,4] - E2[4,38,42] <= 1
 twoApartNights[38,42,5]: X[38,5] + X[42,5] - E2[5,38,42] <= 1
 twoApartNights[38,42,6]: X[38,6] + X[42,6] - E2[6,38,42] <= 1
 twoApartNights[38,42,7]: X[38,7] + X[42,7] - E2[7,38,42] <= 1
 twoApartNights[38,42,8]: X[38,8] + X[42,8] - E2[8,38,42] <= 1
 twoApartNights[38,42,9]: X[38,9] + X[42,9] - E2[9,38,42] <= 1
 twoApartNights[38,42,10]: X[38,10] + X[42,10] - E2[10,38,42] <= 1
 twoApartNights[38,42,11]: X[38,11] + X[42,11] - E2[11,38,42] <= 1
 twoApartNights[38,42,12]: X[38,12] + X[42,12] - E2[12,38,42] <= 1
 twoApartNights[38,42,13]: X[38,13] + X[42,13] - E2[13,38,42] <= 1
 twoApartNights[12,16,1]: X[12,1] + X[16,1] - E2[1,12,16] <= 1
 twoApartNights[12,16,2]: X[12,2] + X[16,2] - E2[2,12,16] <= 1
 twoApartNights[12,16,3]: X[12,3] + X[16,3] - E2[3,12,16] <= 1
 twoApartNights[12,16,4]: X[12,4] + X[16,4] - E2[4,12,16] <= 1
 twoApartNights[12,16,5]: X[12,5] + X[16,5] - E2[5,12,16] <= 1
 twoApartNights[12,16,6]: X[12,6] + X[16,6] - E2[6,12,16] <= 1
 twoApartNights[12,16,7]: X[12,7] + X[16,7] - E2[7,12,16] <= 1
 twoApartNights[12,16,8]: X[12,8] + X[16,8] - E2[8,12,16] <= 1
 twoApartNights[12,16,9]: X[12,9] + X[16,9] - E2[9,12,16] <= 1
 twoApartNights[12,16,10]: X[12,10] + X[16,10] - E2[10,12,16] <= 1
 twoApartNights[12,16,11]: X[12,11] + X[16,11] - E2[11,12,16] <= 1
 twoApartNights[12,16,12]: X[12,12] + X[16,12] - E2[12,12,16] <= 1
 twoApartNights[12,16,13]: X[12,13] + X[16,13] - E2[13,12,16] <= 1
 twoApartNights[26,30,1]: X[26,1] + X[30,1] - E2[1,26,30] <= 1
 twoApartNights[26,30,2]: X[26,2] + X[30,2] - E2[2,26,30] <= 1
 twoApartNights[26,30,3]: X[26,3] + X[30,3] - E2[3,26,30] <= 1
 twoApartNights[26,30,4]: X[26,4] + X[30,4] - E2[4,26,30] <= 1
 twoApartNights[26,30,5]: X[26,5] + X[30,5] - E2[5,26,30] <= 1
 twoApartNights[26,30,6]: X[26,6] + X[30,6] - E2[6,26,30] <= 1
 twoApartNights[26,30,7]: X[26,7] + X[30,7] - E2[7,26,30] <= 1
 twoApartNights[26,30,8]: X[26,8] + X[30,8] - E2[8,26,30] <= 1
 twoApartNights[26,30,9]: X[26,9] + X[30,9] - E2[9,26,30] <= 1
 twoApartNights[26,30,10]: X[26,10] + X[30,10] - E2[10,26,30] <= 1
 twoApartNights[26,30,11]: X[26,11] + X[30,11] - E2[11,26,30] <= 1
 twoApartNights[26,30,12]: X[26,12] + X[30,12] - E2[12,26,30] <= 1
 twoApartNights[26,30,13]: X[26,13] + X[30,13] - E2[13,26,30] <= 1
 nightExcess[1,1]: X[2,1] + X[4,1] + X[6,1] + X[8,1] + X[10,1] + X[12,1]
   + X[14,1] - E3[1,1] <= 2
 nightExcess[1,2]: X[2,2] + X[4,2] + X[6,2] + X[8,2] + X[10,2] + X[12,2]
   + X[14,2] - E3[1,2] <= 2
 nightExcess[1,3]: X[2,3] + X[4,3] + X[6,3] + X[8,3] + X[10,3] + X[12,3]
   + X[14,3] - E3[1,3] <= 2
 nightExcess[1,4]: X[2,4] + X[4,4] + X[6,4] + X[8,4] + X[10,4] + X[12,4]
   + X[14,4] - E3[1,4] <= 2
 nightExcess[1,5]: X[2,5] + X[4,5] + X[6,5] + X[8,5] + X[10,5] + X[12,5]
   + X[14,5] - E3[1,5] <= 2
 nightExcess[1,6]: X[2,6] + X[4,6] + X[6,6] + X[8,6] + X[10,6] + X[12,6]
   + X[14,6] - E3[1,6] <= 2
 nightExcess[1,7]: X[2,7] + X[4,7] + X[6,7] + X[8,7] + X[10,7] + X[12,7]
   + X[14,7] - E3[1,7] <= 2
 nightExcess[1,8]: X[2,8] + X[4,8] + X[6,8] + X[8,8] + X[10,8] + X[12,8]
   + X[14,8] - E3[1,8] <= 2
 nightExcess[1,9]: X[2,9] + X[4,9] + X[6,9] + X[8,9] + X[10,9] + X[12,9]
   + X[14,9] - E3[1,9] <= 2
 nightExcess[1,10]: X[2,10] + X[4,10] + X[6,10] + X[8,10] + X[10,10]
   + X[12,10] + X[14,10] - E3[1,10] <= 2
 nightExcess[1,11]: X[2,11] + X[4,11] + X[6,11] + X[8,11] + X[10,11]
   + X[12,11] + X[14,11] - E3[1,11] <= 2
 nightExcess[1,12]: X[2,12] + X[4,12] + X[6,12] + X[8,12] + X[10,12]
   + X[12,12] + X[14,12] - E3[1,12] <= 2
 nightExcess[1,13]: X[2,13] + X[4,13] + X[6,13] + X[8,13] + X[10,13]
   + X[12,13] + X[14,13] - E3[1,13] <= 2
 nightExcess[2,1]: X[16,1] + X[18,1] + X[20,1] + X[22,1] + X[24,1]
   + X[26,1] + X[28,1] - E3[2,1] <= 2
 nightExcess[2,2]: X[16,2] + X[18,2] + X[20,2] + X[22,2] + X[24,2]
   + X[26,2] + X[28,2] - E3[2,2] <= 2
 nightExcess[2,3]: X[16,3] + X[18,3] + X[20,3] + X[22,3] + X[24,3]
   + X[26,3] + X[28,3] - E3[2,3] <= 2
 nightExcess[2,4]: X[16,4] + X[18,4] + X[20,4] + X[22,4] + X[24,4]
   + X[26,4] + X[28,4] - E3[2,4] <= 2
 nightExcess[2,5]: X[16,5] + X[18,5] + X[20,5] + X[22,5] + X[24,5]
   + X[26,5] + X[28,5] - E3[2,5] <= 2
 nightExcess[2,6]: X[16,6] + X[18,6] + X[20,6] + X[22,6] + X[24,6]
   + X[26,6] + X[28,6] - E3[2,6] <= 2
 nightExcess[2,7]: X[16,7] + X[18,7] + X[20,7] + X[22,7] + X[24,7]
   + X[26,7] + X[28,7] - E3[2,7] <= 2
 nightExcess[2,8]: X[16,8] + X[18,8] + X[20,8] + X[22,8] + X[24,8]
   + X[26,8] + X[28,8] - E3[2,8] <= 2
 nightExcess[2,9]: X[16,9] + X[18,9] + X[20,9] + X[22,9] + X[24,9]
   + X[26,9] + X[28,9] - E3[2,9] <= 2
 nightExcess[2,10]: X[16,10] + X[18,10] + X[20,10] + X[22,10] + X[24,10]
   + X[26,10] + X[28,10] - E3[2,10] <= 2
 nightExcess[2,11]: X[16,11] + X[18,11] + X[20,11] + X[22,11] + X[24,11]
   + X[26,11] + X[28,11] - E3[2,11] <= 2
 nightExcess[2,12]: X[16,12] + X[18,12] + X[20,12] + X[22,12] + X[24,12]
   + X[26,12] + X[28,12] - E3[2,12] <= 2
 nightExcess[2,13]: X[16,13] + X[18,13] + X[20,13] + X[22,13] + X[24,13]
   + X[26,13] + X[28,13] - E3[2,13] <= 2
 nightExcess[3,1]: X[30,1] + X[32,1] + X[34,1] + X[36,1] + X[38,1]
   + X[40,1] + X[42,1] - E3[3,1] <= 2
 nightExcess[3,2]: X[30,2] + X[32,2] + X[34,2] + X[36,2] + X[38,2]
   + X[40,2] + X[42,2] - E3[3,2] <= 2
 nightExcess[3,3]: X[30,3] + X[32,3] + X[34,3] + X[36,3] + X[38,3]
   + X[40,3] + X[42,3] - E3[3,3] <= 2
 nightExcess[3,4]: X[30,4] + X[32,4] + X[34,4] + X[36,4] + X[38,4]
   + X[40,4] + X[42,4] - E3[3,4] <= 2
 nightExcess[3,5]: X[30,5] + X[32,5] + X[34,5] + X[36,5] + X[38,5]
   + X[40,5] + X[42,5] - E3[3,5] <= 2
 nightExcess[3,6]: X[30,6] + X[32,6] + X[34,6] + X[36,6] + X[38,6]
   + X[40,6] + X[42,6] - E3[3,6] <= 2
 nightExcess[3,7]: X[30,7] + X[32,7] + X[34,7] + X[36,7] + X[38,7]
   + X[40,7] + X[42,7] - E3[3,7] <= 2
 nightExcess[3,8]: X[30,8] + X[32,8] + X[34,8] + X[36,8] + X[38,8]
   + X[40,8] + X[42,8] - E3[3,8] <= 2
 nightExcess[3,9]: X[30,9] + X[32,9] + X[34,9] + X[36,9] + X[38,9]
   + X[40,9] + X[42,9] - E3[3,9] <= 2
 nightExcess[3,10]: X[30,10] + X[32,10] + X[34,10] + X[36,10] + X[38,10]
   + X[40,10] + X[42,10] - E3[3,10] <= 2
 nightExcess[3,11]: X[30,11] + X[32,11] + X[34,11] + X[36,11] + X[38,11]
   + X[40,11] + X[42,11] - E3[3,11] <= 2
 nightExcess[3,12]: X[30,12] + X[32,12] + X[34,12] + X[36,12] + X[38,12]
   + X[40,12] + X[42,12] - E3[3,12] <= 2
 nightExcess[3,13]: X[30,13] + X[32,13] + X[34,13] + X[36,13] + X[38,13]
   + X[40,13] + X[42,13] - E3[3,13] <= 2
 dayNightSplit[1]: X[1,1] + X[3,1] + X[5,1] + X[7,1] + X[9,1] + X[11,1]
   + X[13,1] + X[15,1] + X[17,1] + X[19,1] + X[21,1] + X[23,1] + X[25,1]
   + X[27,1] + X[29,1] + X[31,1] + X[33,1] + X[35,1] + X[37,1] + X[39,1]
   + X[41,1] - Dplus[1] + Dminus[1] = 2
 dayNightSplit[2]: X[1,2] + X[3,2] + X[5,2] + X[7,2] + X[9,2] + X[11,2]
   + X[13,2] + X[15,2] + X[17,2] + X[19,2] + X[21,2] + X[23,2] + X[25,2]
   + X[27,2] + X[29,2] + X[31,2] + X[33,2] + X[35,2] + X[37,2] + X[39,2]
   + X[41,2] - Dplus[2] + Dminus[2] = 1
 dayNightSplit[3]: X[1,3] + X[3,3] + X[5,3] + X[7,3] + X[9,3] + X[11,3]
   + X[13,3] + X[15,3] + X[17,3] + X[19,3] + X[21,3] + X[23,3] + X[25,3]
   + X[27,3] + X[29,3] + X[31,3] + X[33,3] + X[35,3] + X[37,3] + X[39,3]
   + X[41,3] - Dplus[3] + Dminus[3] = 2
 dayNightSplit[4]: X[1,4] + X[3,4] + X[5,4] + X[7,4] + X[9,4] + X[11,4]
   + X[13,4] + X[15,4] + X[17,4] + X[19,4] + X[21,4] + X[23,4] + X[25,4]
   + X[27,4] + X[29,4] + X[31,4] + X[33,4] + X[35,4] + X[37,4] + X[39,4]
   + X[41,4] - Dplus[4] + Dminus[4] = 1
 dayNightSplit[5]: X[1,5] + X[3,5] + X[5,5] + X[7,5] + X[9,5] + X[11,5]
   + X[13,5] + X[15,5] + X[17,5] + X[19,5] + X[21,5] + X[23,5] + X[25,5]
   + X[27,5] + X[29,5] + X[31,5] + X[33,5] + X[35,5] + X[37,5] + X[39,5]
   + X[41,5] - Dplus[5] + Dminus[5] = 1
 dayNightSplit[6]: X[1,6] + X[3,6] + X[5,6] + X[7,6] + X[9,6] + X[11,6]
   + X[13,6] + X[15,6] + X[17,6] + X[19,6] + X[21,6] + X[23,6] + X[25,6]
   + X[27,6] + X[29,6] + X[31,6] + X[33,6] + X[35,6] + X[37,6] + X[39,6]
   + X[41,6] - Dplus[6] + Dminus[6] = 1
 dayNightSplit[7]: X[1,7] + X[3,7] + X[5,7] + X[7,7] + X[9,7] + X[11,7]
   + X[13,7] + X[15,7] + X[17,7] + X[19,7] + X[21,7] + X[23,7] + X[25,7]
   + X[27,7] + X[29,7] + X[31,7] + X[33,7] + X[35,7] + X[37,7] + X[39,7]
   + X[41,7] - Dplus[7] + Dminus[7] = 1
 dayNightSplit[8]: X[1,8] + X[3,8] + X[5,8] + X[7,8] + X[9,8] + X[11,8]
   + X[13,8] + X[15,8] + X[17,8] + X[19,8] + X[21,8] + X[23,8] + X[25,8]
   + X[27,8] + X[29,8] + X[31,8] + X[33,8] + X[35,8] + X[37,8] + X[39,8]
   + X[41,8] - Dplus[8] + Dminus[8] = 1
 dayNightSplit[9]: X[1,9] + X[3,9] + X[5,9] + X[7,9] + X[9,9] + X[11,9]
   + X[13,9] + X[15,9] + X[17,9] + X[19,9] + X[21,9] + X[23,9] + X[25,9]
   + X[27,9] + X[29,9] + X[31,9] + X[33,9] + X[35,9] + X[37,9] + X[39,9]
   + X[41,9] - Dplus[9] + Dminus[9] = 2
 dayNightSplit[10]: X[1,10] + X[3,10] + X[5,10] + X[7,10] + X[9,10]
   + X[11,10] + X[13,10] + X[15,10] + X[17,10] + X[19,10] + X[21,10]
   + X[23,10] + X[25,10] + X[27,10] + X[29,10] + X[31,10] + X[33,10]
   + X[35,10] + X[37,10] + X[39,10] + X[41,10] - Dplus[10] + Dminus[10]
   = 1
 dayNightSplit[11]: X[1,11] + X[3,11] + X[5,11] + X[7,11] + X[9,11]
   + X[11,11] + X[13,11] + X[15,11] + X[17,11] + X[19,11] + X[21,11]
   + X[23,11] + X[25,11] + X[27,11] + X[29,11] + X[31,11] + X[33,11]
   + X[35,11] + X[37,11] + X[39,11] + X[41,11] - Dplus[11] + Dminus[11]
   = 2
 dayNightSplit[12]: X[1,12] + X[3,12] + X[5,12] + X[7,12] + X[9,12]
   + X[11,12] + X[13,12] + X[15,12] + X[17,12] + X[19,12] + X[21,12]
   + X[23,12] + X[25,12] + X[27,12] + X[29,12] + X[31,12] + X[33,12]
   + X[35,12] + X[37,12] + X[39,12] + X[41,12] - Dplus[12] + Dminus[12]
   = 1
 dayNightSplit[13]: X[1,13] + X[3,13] + X[5,13] + X[7,13] + X[9,13]
   + X[11,13] + X[13,13] + X[15,13] + X[17,13] + X[19,13] + X[21,13]
   + X[23,13] + X[25,13] + X[27,13] + X[29,13] + X[31,13] + X[33,13]
   + X[35,13] + X[37,13] + X[39,13] + X[41,13] - Dplus[13] + Dminus[13]
   = 1
Bounds
Binaries
 X[1,1] X[1,2] X[1,3] X[1,4] X[1,5] X[1,6] X[1,7] X[1,8] X[1,9] X[1,10]
 X[1,11] X[1,12] X[1,13] X[2,1] X[2,2] X[2,3] X[2,4] X[2,5] X[2,6] X[2,7]
 X[2,8] X[2,9] X[2,10] X[2,11] X[2,12] X[2,13] X[3,1] X[3,2] X[3,3] X[3,4]
 X[3,5] X[3,6] X[3,7] X[3,8] X[3,9] X[3,10] X[3,11] X[3,12] X[3,13] X[4,1]
 X[4,2] X[4,3] X[4,4] X[4,5] X[4,6] X[4,7] X[4,8] X[4,9] X[4,10] X[4,11]
 X[4,12] X[4,13] X[5,1] X[5,2] X[5,3] X[5,4] X[5,5] X[5,6] X[5,7] X[5,8]
 X[5,9] X[5,10] X[5,11] X[5,12] X[5,13] X[6,1] X[6,2] X[6,3] X[6,4] X[6,5]
 X[6,6] X[6,7] X[6,8] X[6,9] X[6,10] X[6,11] X[6,12] X[6,13] X[7,1] X[7,2]
 X[7,3] X[7,4] X[7,5] X[7,6] X[7,7] X[7,8] X[7,9] X[7,10] X[7,11] X[7,12]
 X[7,13] X[8,1] X[8,2] X[8,3] X[8,4] X[8,5] X[8,6] X[8,7] X[8,8] X[8,9]
 X[8,10] X[8,11] X[8,12] X[8,13] X[9,1] X[9,2] X[9,3] X[9,4] X[9,5] X[9,6]
 X[9,7] X[9,8] X[9,9] X[9,10] X[9,11] X[9,12] X[9,13] X[10,1] X[10,2]
 X[10,3] X[10,4] X[10,5] X[10,6] X[10,7] X[10,8] X[10,9] X[10,10] X[10,11]
 X[10,12] X[10,13] X[11,1] X[11,2] X[11,3] X[11,4] X[11,5] X[11,6] X[11,7]
 X[11,8] X[11,9] X[11,10] X[11,11] X[11,12] X[11,13] X[12,1] X[12,2]
 X[12,3] X[12,4] X[12,5] X[12,6] X[12,7] X[12,8] X[12,9] X[12,10] X[12,11]
 X[12,12] X[12,13] X[13,1] X[13,2] X[13,3] X[13,4] X[13,5] X[13,6] X[13,7]
 X[13,8] X[13,9] X[13,10] X[13,11] X[13,12] X[13,13] X[14,1] X[14,2]
 X[14,3] X[14,4] X[14,5] X[14,6] X[14,7] X[14,8] X[14,9] X[14,10] X[14,11]
 X[14,12] X[14,13] X[15,1] X[15,2] X[15,3] X[15,4] X[15,5] X[15,6] X[15,7]
 X[15,8] X[15,9] X[15,10] X[15,11] X[15,12] X[15,13] X[16,1] X[16,2]
 X[16,3] X[16,4] X[16,5] X[16,6] X[16,7] X[16,8] X[16,9] X[16,10] X[16,11]
 X[16,12] X[16,13] X[17,1] X[17,2] X[17,3] X[17,4] X[17,5] X[17,6] X[17,7]
 X[17,8] X[17,9] X[17,10] X[17,11] X[17,12] X[17,13] X[18,1] X[18,2]
 X[18,3] X[18,4] X[18,5] X[18,6] X[18,7] X[18,8] X[18,9] X[18,10] X[18,11]
 X[18,12] X[18,13] X[19,1] X[19,2] X[19,3] X[19,4] X[19,5] X[19,6] X[19,7]
 X[19,8] X[19,9] X[19,10] X[19,11] X[19,12] X[19,13] X[20,1] X[20,2]
 X[20,3] X[20,4] X[20,5] X[20,6] X[20,7] X[20,8] X[20,9] X[20,10] X[20,11]
 X[20,12] X[20,13] X[21,1] X[21,2] X[21,3] X[21,4] X[21,5] X[21,6] X[21,7]
 X[21,8] X[21,9] X[21,10] X[21,11] X[21,12] X[21,13] X[22,1] X[22,2]
 X[22,3] X[22,4] X[22,5] X[22,6] X[22,7] X[22,8] X[22,9] X[22,10] X[22,11]
 X[22,12] X[22,13] X[23,1] X[23,2] X[23,3] X[23,4] X[23,5] X[23,6] X[23,7]
 X[23,8] X[23,9] X[23,10] X[23,11] X[23,12] X[23,13] X[24,1] X[24,2]
 X[24,3] X[24,4] X[24,5] X[24,6] X[24,7] X[24,8] X[24,9] X[24,10] X[24,11]
 X[24,12] X[24,13] X[25,1] X[25,2] X[25,3] X[25,4] X[25,5] X[25,6] X[25,7]
 X[25,8] X[25,9] X[25,10] X[25,11] X[25,12] X[25,13] X[26,1] X[26,2]
 X[26,3] X[26,4] X[26,5] X[26,6] X[26,7] X[26,8] X[26,9] X[26,10] X[26,11]
 X[26,12] X[26,13] X[27,1] X[27,2] X[27,3] X[27,4] X[27,5] X[27,6] X[27,7]
 X[27,8] X[27,9] X[27,10] X[27,11] X[27,12] X[27,13] X[28,1] X[28,2]
 X[28,3] X[28,4] X[28,5] X[28,6] X[28,7] X[28,8] X[28,9] X[28,10] X[28,11]
 X[28,12] X[28,13] X[29,1] X[29,2] X[29,3] X[29,4] X[29,5] X[29,6] X[29,7]
 X[29,8] X[29,9] X[29,10] X[29,11] X[29,12] X[29,13] X[30,1] X[30,2]
 X[30,3] X[30,4] X[30,5] X[30,6] X[30,7] X[30,8] X[30,9] X[30,10] X[30,11]
 X[30,12] X[30,13] X[31,1] X[31,2] X[31,3] X[31,4] X[31,5] X[31,6] X[31,7]
 X[31,8] X[31,9] X[31,10] X[31,11] X[31,12] X[31,13] X[32,1] X[32,2]
 X[32,3] X[32,4] X[32,5] X[32,6] X[32,7] X[32,8] X[32,9] X[32,10] X[32,11]
 X[32,12] X[32,13] X[33,1] X[33,2] X[33,3] X[33,4] X[33,5] X[33,6] X[33,7]
 X[33,8] X[33,9] X[33,10] X[33,11] X[33,12] X[33,13] X[34,1] X[34,2]
 X[34,3] X[34,4] X[34,5] X[34,6] X[34,7] X[34,8] X[34,9] X[34,10] X[34,11]
 X[34,12] X[34,13] X[35,1] X[35,2] X[35,3] X[35,4] X[35,5] X[35,6] X[35,7]
 X[35,8] X[35,9] X[35,10] X[35,11] X[35,12] X[35,13] X[36,1] X[36,2]
 X[36,3] X[36,4] X[36,5] X[36,6] X[36,7] X[36,8] X[36,9] X[36,10] X[36,11]
 X[36,12] X[36,13] X[37,1] X[37,2] X[37,3] X[37,4] X[37,5] X[37,6] X[37,7]
 X[37,8] X[37,9] X[37,10] X[37,11] X[37,12] X[37,13] X[38,1] X[38,2]
 X[38,3] X[38,4] X[38,5] X[38,6] X[38,7] X[38,8] X[38,9] X[38,10] X[38,11]
 X[38,12] X[38,13] X[39,1] X[39,2] X[39,3] X[39,4] X[39,5] X[39,6] X[39,7]
 X[39,8] X[39,9] X[39,10] X[39,11] X[39,12] X[39,13] X[40,1] X[40,2]
 X[40,3] X[40,4] X[40,5] X[40,6] X[40,7] X[40,8] X[40,9] X[40,10] X[40,11]
 X[40,12] X[40,13] X[41,1] X[41,2] X[41,3] X[41,4] X[41,5] X[41,6] X[41,7]
 X[41,8] X[41,9] X[41,10] X[41,11] X[41,12] X[41,13] X[42,1] X[42,2]
 X[42,3] X[42,4] X[42,5] X[42,6] X[42,7] X[42,8] X[42,9] X[42,10] X[42,11]
 X[42,12] X[42,13] Y[1,1] Y[1,2] Y[1,3] Y[1,4] Y[1,5] Y[1,6] Y[1,7] Y[1,8]
 Y[1,9] Y[1,10] Y[1,11] Y[1,12] Y[1,13] Y[2,1] Y[2,2] Y[2,3] Y[2,4] Y[2,5]
 Y[2,6] Y[2,7] Y[2,8] Y[2,9] Y[2,10] Y[2,11] Y[2,12] Y[2,13] Y[3,1] Y[3,2]
 Y[3,3] Y[3,4] Y[3,5] Y[3,6] Y[3,7] Y[3,8] Y[3,9] Y[3,10] Y[3,11] Y[3,12]
 Y[3,13] Z[1,1] Z[1,2] Z[1,3] Z[1,4] Z[1,5] Z[1,6] Z[1,7] Z[1,8] Z[1,9]
 Z[1,10] Z[1,11] Z[1,12] Z[1,13] Z[2,1] Z[2,2] Z[2,3] Z[2,4] Z[2,5] Z[2,6]
 Z[2,7] Z[2,8] Z[2,9] Z[2,10] Z[2,11] Z[2,12] Z[2,13] Z[3,1] Z[3,2] Z[3,3]
 Z[3,4] Z[3,5] Z[3,6] Z[3,7] Z[3,8] Z[3,9] Z[3,10] Z[3,11] Z[3,12] Z[3,13]
End
