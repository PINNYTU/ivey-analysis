\ Model Jaikumar with contract
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  91 X[15] + 50 X[84] + 470000 Y
Subject To
 spinning: 0.0333333333333333 X[15] + 0.0166666666666667 X[84]
   + 166.6666666666667 Y <= 300
 drawtwisting: 0.0666666666666667 X[15] + 0.0833333333333333 X[84]
   + 333.3333333333333 Y <= 1600
 min_kg_15: X[15] + 5000 Y >= 3000
 min_kg_84: X[84] >= 6000
 max_kg_15: X[15] + 5000 Y <= 10000
 max_kg_84: X[84] <= 15000
Bounds
Binaries
 Y
End
