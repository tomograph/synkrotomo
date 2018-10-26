import "futlib/array"

module sirtLIB = {
  let vecADD [vct_len] (vct1 : [vct_len]f32) (vct2 : [vct_len]f32) : [vct_len]f32 =
  map2 (+) vct1 vct2

  let sirtVecVec [m] [q]
                  (matRow : [m]f32)
                  (vect : [q]f32)
                  (num_cols : i32) =
    map (\j ->
      unsafe (reduce (+) 0 <| map (\k ->
        unsafe (matRow[k] * vect[j*m+k])
        ) (iota m)
      )) (iota num_cols)

  let sparseMatMult (mat_vals:[][](f32,i32)) vect num_cols num_rows =
    map (\row ->
      map (\(v, ind) ->
        reduce (+) 0 (map (\i ->
          if ind == -1 then 0.0
          else unsafe (v * vect[i*num_cols + ind])
        ) (iota ((length vect)/num_cols)) )
      ) row
    ) mat_vals

}
