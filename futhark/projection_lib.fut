import "matrix_lib"
open Matrix

module Projection = {

     --segmented scan with (+) on floats:
     let sgmSumF32 [n]
           (flg : [n]i32)
           (arr : [n]f32) : [n]f32 =
               let flgs_vals =
                scan ( \ (f1, x1) (f2,x2) ->
                        let f = f1 | f2 in
                        if f2 > 0 then (f, x2)
                        else (f, x1 + x2) )
                     (0,0.0f32) (zip flg arr)
               let (_, vals) = unzip flgs_vals
               in vals

     -- segmented scan with (+) on ints:
     let sgmSumI32 [n]
           (flg : [n]i32)
           (arr : [n]i32) : [n]i32 =
               let flgs_vals =
                scan ( \ (i1, x1) (i2,x2) ->
                        let i = i1 | i2 in
                        if i2 > 0 then (i, x2)
                        else (i, x1 + x2) )
                     (0,0i32) (zip flg arr)
               let (_, vals) = unzip flgs_vals
               in vals

      -- unzip for 2D arrays
      let unzip_d2 (xs: [][](f32,i32)): ([][]f32,[][]i32) =
          xs |> map unzip
               |> unzip

      -- step in radix sort
      let rsort_step [n] (xs: [n](f32, i32, i32), bitn: i32): [n](f32, i32, i32) =
                let (data,rays,pixels) = unzip3 xs
                let unsigned = map(\p -> u32.i32 p) pixels
                let bits1 = map (\x -> (i32.u32 (x >> u32.i32 bitn)) & 1) unsigned
                let bits0 = map (1-) bits1
                let idxs0 = map2 (*) bits0 (scan (+) 0 bits0)
                let idxs1 = scan (+) 0 bits1
                let offs  = reduce (+) 0 bits0
                let idxs1 = map2 (*) bits1 (map (+offs) idxs1)
                let idxs  = map2 (+) idxs0 idxs1
                let idxs  = map (\x->x-1) idxs
                in scatter (copy xs) idxs xs

      -- Radix sort algorithm, ascending
      let rsort [n] (xs: [n](f32, i32, i32)): [n](f32,i32,i32) =
               loop (xs) for i < 32 do rsort_step(xs,i)

      -- sparse matrix vector multiplication
      let spMatVctMult [num_elms] [vct_len] [num_rows]
          (mat_val : [num_elms](i32,f32))
          (shp_scn : [num_rows]i32)
          (vct : [vct_len]f32) : [num_rows]f32 =
               let len = shp_scn[num_rows-1]
               let shp_inds =
                   map (\i -> if i==0 then 0
                     else unsafe shp_scn[i-1]
                   ) (iota num_rows)
               let flags = scatter ( replicate len 0)
                     shp_inds ( replicate num_rows 1)
               let prods = map (\(i,x) -> x*(unsafe vct[i])) mat_val
               let sums = sgmSumF32 flags prods
               let mat_inds = map (\i -> i-1) shp_scn
               in map (\i -> unsafe sums[i]) mat_inds

      let notSparseMatMult [num_rows] [num_cols]
                           (mat_vals : [num_rows][num_cols](f32,i32))
                           (vect : []f32) : [num_rows]f32 =
          map (\row -> reduce (+) 0 <| map (\(v, ind) -> unsafe (if ind == -1 then 0.0 else v*vect[ind]) ) row ) mat_vals

      -- not sparse, nested, matrix vector multiplication for backprojection, uses a padded version of the matrix
      let notSparseMatMult_back [num_rows] [num_cols]
                            (mat_vals : [num_rows][num_cols]f32)
                            (vect : []f32) : [num_rows]f32 =
          map (\row -> reduce (+) 0 <| map2 (*) row vect ) mat_vals

     -- pads and transposes the matrix, nested, will perform better when tiling is improved in futhark
     let trans_map [m] [n]
                    (matrix : [m][n](f32, i32)) (gridsize: i32): [][]f32 =
               let rs = gridsize*gridsize
               let padded = map (\row -> let (vals, inds) = (unzip row) in scatter (replicate rs 0.0f32) inds vals ) matrix
               in transpose padded

     -- let inverseRowSum [num_rows] [num_cols]
     --                      (mat_vals : [num_rows][num_cols](f32,i32)) : [num_rows]f32 =
     --     map (\row -> 1/(reduce (+) 0 <| map (\(v, ind) -> unsafe (if ind == -1 then 0.0 else v) ) row )) mat_vals

     let backprojection  [a][r][p](angles : [a]f32)
                         (rhos : [r]f32)
                         (projections : [p]f32)
                         (gridsize: i32)
                         (stepSize : i32) : []f32=
               let halfsize = r32(gridsize)/2
               let entryexitpoints =  convert2entryexit angles rhos halfsize
               let runLen = (p/stepSize)
               -- result array
               let backmat = replicate (gridsize*gridsize) 0.0f32
               -- stripmined, sequential outer loop, mapped inner
               let (backmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entryexitpoints, p) = (backmat, 0, runLen, stepSize, gridsize, entryexitpoints, p)
                   while ( run < runLen ) do
                       -- if the number of entrypoints doesn't line perfectly up with the stepsize
                       let step = if (run+1)*stepSize >= p then p - run*stepSize else stepSize
                       -- calc part of matrix, stepSize rows
                       let halfgridsize = gridsize/2

                       let partmatrix = map(\(ent,ext) -> (flatten(map (\i ->
                                 calculate_weight ent ext i gridsize
                            )((-halfgridsize)...(halfgridsize-1))))) (entryexitpoints[(run*stepSize) : (run*stepSize + step)])

                       -- transpose
                       let transmat = trans_map partmatrix gridsize
                       in ((map2 (+) ((notSparseMatMult_back transmat projections[(run*stepSize) : (run*stepSize + step)])) output), run+1, runLen, stepSize, gridsize, entryexitpoints, p)
               in backmat

     let SIRT [a][r][n](angles: [a]f32) (rhos: [r]f32) (img: [n][n]f32) (projections: []f32) (iterations: i32): [n][n]f32 =
          let res = loop (img) = (img) for iter < iterations do
               let diff = projection_difference angles rhos img projections
               -- use stepsize = n for now (i.e one angle at a time)
               let bp = backprojection angles rhos projections n n
               in (map(\i -> (unsafe (map2 (+) (unsafe img[i]) (unsafe bp[i*n:(i+1)*n]))))(iota n))
          in res
}
