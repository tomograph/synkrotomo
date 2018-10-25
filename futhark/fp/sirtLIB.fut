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

}
