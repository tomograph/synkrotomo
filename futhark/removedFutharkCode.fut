-- let transposeNF [num_elms] [num_rows]
--                  (mat_val : [num_elms](i32,f32))
--                  (mat_shp : [num_rows]i32) : [num_elms]f32 =
--
--   let inds =

let transposeSparseFlatMat [num_elms]
                            (mat_vals : [num_elms](i32,f32))
                            (flag : [num_elms]i32)
                            (num_cols : i32) : [num_elms](i32,f32) =
  let (cs, vs) = unzip mat_vals
  let rs = scan (+) 0 flag
  let mat = zip3 cs rs vs
  let transposed = map (\i -> (filter (\(c, r, v) -> c == i) mat)) iota num_cols
  let flattrans = flatten transposed
  let (cs, rs, vs) = unzip3 flattrans
  in zip rs vs


let test m:[]i32 : []i32 = transp(m)
let transposeFlatMatNested [num_elms] [num_rows]
                      (mat_val : [num_elms](i32,f32))
                      (flag : [num_elms]i32)
                      (mat_shp : [num_rows]i32)
                      (num_cols : i32)  =
  -- let emptyMat = replicate num_elms 0
  let rows = scan (+) 0 flag
  let (row_inds, vals) = unzip mat_val
  let newMatVals = zip rows vals
  let new_inds = map (\i -> unsafe (reduce(+) 0 <| map (\j -> unsafe (if j < 0 then 0 else 1)) (filter (== i) row_inds)) ) (iota num_cols)

  let new_shp = map (\i -> unsafe (reduce(+) 0 <| map (\j -> unsafe (if j < 0 then 0 else 1)) (filter (== i) row_inds)) ) (iota num_cols)
  -- let testval =
  -- let test = map(\i -> trace i+0) new_shp
  let shp_sc = scan (+) 0 new_shp
  let shp_ind = map (\i -> unsafe (if i == 0 then 0 else shp_sc[i-1])) (iota 2)
  let newflag = scatter (replicate num_elms 0) shp_ind (replicate num_cols 1)




  map (\i -> Lambda-function) (zip row_inds, newMatVals)
  -- let transMat = map (\i -> (map (\((ind, v), r) -> if ind == i then ((r, v), ind) ) (zip mat_val rows)) ) (iota num_cols)
  -- let (mat, _) = unzip transMat
  -- let (res, _) unzip mat
  -- in res
