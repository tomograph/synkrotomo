import "futlib/array"

module sirtLIB = {
  let vecADD [vct_len] (vct1 : [vct_len]f32) (vct2 : [vct_len]f32) : [vct_len]f32 =
  map2 (+) vct1 vct2

  -- let transposeSparseFlatMat [num_elms]
  --                             (mat_vals : [num_elms](i32,f32))
  --                             (flag : [num_elms]i32)
  --                             (num_cols : i32) : [num_elms](i32,f32) =
  --   let (cs, vs) = unzip mat_vals
  --   let rs = scan (+) 0 flag
  --   let mat = zip3 cs rs vs
  --   let transposed = map (\i -> (filter (\(c, r, v) -> c == i) mat)) (iota num_cols)
  --   let flattrans = flatten transposed
  --   let (cs, rs, vs) = unzip3 flattrans
  --   in zip rs vs

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
      -- map (\i -> Lambda-function) (iota ((length vect)/num_cols))

      map (\(v, ind) -> --trace ind --reduce (+) 0 <|
        reduce (+) 0 (map (\i ->
      -- if ind > -1 then trace (v * vect[ind]) else 0

          if ind == -1 then 0.0
          else unsafe (v * vect[i*num_cols + ind])
        ) (iota ((length vect)/num_cols)) )
      ) row
    ) mat_vals

  -- let handleMat mat_vals num_cols num_rows =
    -- let (vals, inds) = unzip mat_vals
    -- let shp = map (\i -> trace (length i)) mat_inds
    -- in map(\i -> trace i+0) shp
     -- map (\i -> trace (length i)) mat_inds
    -- map (\i -> length i) mat_vals
    -- map (\i -> map (\j -> trace ( mat_vals.1[i][j] ) ) (iota num_cols) ) (iota num_rows)

}
