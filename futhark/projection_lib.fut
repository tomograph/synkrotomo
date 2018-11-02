import "futlib/array"
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

     -- gets the shape of a matrix - i.e number of entries pr. row when mat is in the format [[(d,i),(d,i)...]] where d is data
     -- and i is column index and -1 means no data
     let getshp (matrix: [][](f32,i32)) : []i32 =
               let dp = unzip_d2 matrix
               let flagsforindexes = map(\r -> map(\p -> if p == -1 then 0 else 1)r)dp.2
               in map(\f -> reduce (+) 0 f)flagsforindexes

     -- helper function to determine if we entered new segment of consecutive numbers in an array.
     let isnewsegment (i: i32) (arr: []i32) : bool =
                         i!=0 && (unsafe arr[i]) != (unsafe arr[i-1])

     let backprojection_semiflat  (angles : []f32)
                         (rays : []f32)
                         (projections : []f32)
                         (gridsize: i32)
                         (stepSize : i32) :[]f32=
               let halfsize = r32(gridsize)/2
               let entrypoints = convert2entry angles rays halfsize
               let totalLen = (length entrypoints)
               let runLen = (totalLen/stepSize)
               -- result array
               let backmat = replicate (gridsize*gridsize) 0.0f32
               -- stripmined, sequential outer loop, mapped inner
               let (backmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entrypoints, totalLen) = (backmat, 0, runLen, stepSize, gridsize, entrypoints, totalLen)
                   while ( run < runLen ) do
                       -- if the number of entrypoints doesn't line perfectly up with the stepsize
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       -- calc part of matrix, stepSize rows
                       let intersections = map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints[(run*stepSize) : (run*stepSize + step)]
                       --convert to triples (data,ray,pixel)
                       let triples_tmp = flatten(map(\i -> map(\v -> (v.1, i, v.2))(unsafe intersections[i])) (iota (length intersections)))
                       -- remove none values
                       let triples = filter (\x -> x.3 != -1) triples_tmp
                       -- sort by pixel indexes
                       let pixelsorted = rsort triples
                       -- slplit int three arrays in order to use pixels only for shp
                       let (data,rays,pixels) = unzip3 pixelsorted
                       -- contains sum of values where a row ends since columns will be rows.
                       let shp_scn_tmp = map (\i -> if (i == step || (isnewsegment i pixels)) then i else 0) (iota (step+1))
                       let shp_scn = filter (\p -> p != 0) shp_scn_tmp
                       let values = map(\x-> (x.2,x.1))pixelsorted
                       let partresult = spMatVctMult values shp_scn projections[(run*stepSize) : (run*stepSize + step)]
                       let result = (map2 (+) partresult output)
                       in (result, run+1, runLen, stepSize, gridsize, entrypoints, totalLen)
               in backmat

     -- pads and transposes the matrix, nested, will perform better when tiling is improved in futhark
     let trans_map [m] [n]
                    (matrix : [m][n](f32, i32)) (gridsize: i32): [][]f32 =
               let rs = gridsize*gridsize
               let padded = map (\row -> let (vals, inds) = (unzip row) in scatter (replicate rs 0.0f32) inds vals ) matrix
               in transpose padded

     -- backprojection nested map version.
     let backprojection_map  (angles : []f32)
                         (rays : []f32)
                         (projections : []f32)
                         (gridsize: i32)
                         (stepSize : i32) : []f32=
               let halfsize = r32(gridsize)/2
               let entrypoints = convert2entry angles rays halfsize
               let totalLen = (length entrypoints)
               let runLen = (totalLen/stepSize)
               -- result array
               let backmat = replicate (gridsize*gridsize) 0.0f32
               -- stripmined, sequential outer loop, mapped inner
               let (backmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entrypoints, totalLen) = (backmat, 0, runLen, stepSize, gridsize, entrypoints, totalLen)
                   while ( run < runLen ) do
                       -- if the number of entrypoints doesn't line perfectly up with the stepsize
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       -- calc part of matrix, stepSize rows
                       let partmatrix = map (\s -> unsafe (lengths_map gridsize (entrypoints[run*stepSize + s].2).1 (entrypoints[run*stepSize + s].2).2 entrypoints[run*stepSize + s].1 )) (iota step)
                       -- transpose
                       let transmat = trans_map partmatrix gridsize
                       -- mult
                       let partresult = (notSparseMatMult_back transmat projections[(run*stepSize) : (run*stepSize + step)])
                       -- add
                       let result = (map2 (+) partresult output)
                       in (result, run+1, runLen, stepSize, gridsize, entrypoints, totalLen)
               in backmat

     let backprojection_jh (angles : []f32)
                         (rays : []f32)
                         (projections : []f32)
                         (gridsize: i32)
                         (stepSize : i32) : []f32=
               let halfsize = r32(gridsize)/2
               let entrypoints = convert2entry angles rays halfsize
               let totalLen = (length entrypoints)
               let runLen = (totalLen/stepSize)
               -- result array
               let backmat = replicate (gridsize*gridsize) 0.0f32
               -- stripmined, sequential outer loop, mapped inner
               let (backmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entrypoints, totalLen) = (backmat, 0, runLen, stepSize, gridsize, entrypoints, totalLen)
                   while ( run < runLen ) do
                       -- if the number of entrypoints doesn't line perfectly up with the stepsize
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       -- calc part of matrix, stepSize rows
                       let partmatrix = map (\s -> unsafe (lengths gridsize (entrypoints[run*stepSize + s].2).1 (entrypoints[run*stepSize + s].2).2 entrypoints[run*stepSize + s].1 )) (iota step)
                       -- transpose
                       let transmat = trans_map partmatrix gridsize
                       -- mult
                       let partresult = (notSparseMatMult_back transmat projections[(run*stepSize) : (run*stepSize + step)])
                       -- add
                       let result = (map2 (+) partresult output)
                       in (result, run+1, runLen, stepSize, gridsize, entrypoints, totalLen)
               in backmat

     let backprojection_doubleparallel  (angles : []f32)
                         (rays : []f32)
                         (projections : []f32)
                         (gridsize: i32)
                         (stepSize : i32) : []f32=
               let halfsize = r32(gridsize)/2
               let entryexitpoints =  convert2entryexit angles rays halfsize
               let totalLen = (length entryexitpoints)
               let runLen = (totalLen/stepSize)
               -- result array
               let backmat = replicate (gridsize*gridsize) 0.0f32
               -- stripmined, sequential outer loop, mapped inner
               let (backmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entryexitpoints, totalLen) = (backmat, 0, runLen, stepSize, gridsize, entryexitpoints, totalLen)
                   while ( run < runLen ) do
                       -- if the number of entrypoints doesn't line perfectly up with the stepsize
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       -- calc part of matrix, stepSize rows
                       let halfgridsize = gridsize/2

                       let partmatrix = map(\(ent,ext) -> (flatten(map (\i ->
                                 calculate_weight ent ext i gridsize
                            )((-halfgridsize)...(halfgridsize-1))))) (entryexitpoints[(run*stepSize) : (run*stepSize + step)])
                       -- transpose
                       let transmat = trans_map partmatrix gridsize
                       -- mult
                       let partresult = (notSparseMatMult_back transmat projections[(run*stepSize) : (run*stepSize + step)])
                       -- add
                       let result = (map2 (+) partresult output)
                       in (result, run+1, runLen, stepSize, gridsize, entryexitpoints, totalLen)
               in backmat

     let forwardprojection_doubleparallel (angles : []f32)
                         (rays : []f32)
                          (voxels : []f32)
                          (stepSize : i32) : []f32 =
               let gridsize = t32(f32.sqrt(r32((length voxels))))
               let halfgridsize = gridsize/2
               let entryexitpoints =  convert2entryexit angles rays (r32(halfgridsize))
               let totalLen =  (length entryexitpoints)
               let runLen = (totalLen/stepSize)
               let testmat = [0f32]
               let (testmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entryexitpoints, totalLen) = (testmat, 0, runLen, stepSize, gridsize, entryexitpoints, totalLen)
                   while ( run < runLen ) do
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       let partmatrix = map(\(ent,ext) -> (flatten(map (\i ->
                                 calculate_weight ent ext i gridsize
                            )(-halfgridsize...halfgridsize-1)))) (entryexitpoints[run*stepSize:run*stepSize+step])
                       let partresult = notSparseMatMult partmatrix voxels
                       in (output++partresult, run+1, runLen, stepSize, gridsize, entryexitpoints, totalLen)
               in (tail testmat)

     let forwardprojection_jh [r][a][n] (angles : [a]f32)
                         (rays : [r]f32)
                          (voxels : [n]f32)
                          (stepSize : i32) : []f32 =
               let gridsize = t32(f32.sqrt(r32((length voxels))))
               let halfsize = r32(gridsize)/2
               let entrypoints = convert2entry angles rays halfsize
               let totalLen = (length entrypoints)
               -- let runLen = if (totalLen/stepSize) == 0 then 1 else (totalLen/stepSize)
               let runLen = (totalLen/stepSize)
               let testmat = [0f32]
               let (testmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entrypoints, totalLen) = (testmat, 0, runLen, stepSize, gridsize, entrypoints, totalLen)
                   while ( run < runLen ) do
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       let partmatrix = map (\s -> unsafe (lengths gridsize (entrypoints[run*stepSize + s].2).1 (entrypoints[run*stepSize + s].2).2 entrypoints[run*stepSize + s].1 )) (iota step)
                       let partresult = notSparseMatMult partmatrix voxels
                       in (output++partresult, run+1, runLen, stepSize, gridsize, entrypoints, totalLen)
               in (tail testmat)

     let forwardprojection_map [r][a][n] (angles : [a]f32)
                         (rays : [r]f32)
                          (voxels : [n]f32)
                          (stepSize : i32) : []f32 =
               let gridsize = t32(f32.sqrt(r32((length voxels))))
               let halfsize = r32(gridsize)/2
               let entrypoints = convert2entry angles rays halfsize
               let totalLen = (length entrypoints)
               let runLen = (totalLen/stepSize)
               let testmat = [0f32]
               -- stripmined, sequential outer loop, mapped inner
               let (testmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entrypoints, totalLen) = (testmat, 0, runLen, stepSize, gridsize, entrypoints, totalLen)
                   while ( run < runLen ) do
                       -- if the number of entrypoints doesn't line perfectly up with the stepsize
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       -- calc part of matrix, stepSize rows
                       let partmatrix = map (\s -> unsafe (lengths_map gridsize (entrypoints[run*stepSize + s].2).1 (entrypoints[run*stepSize + s].2).2 entrypoints[run*stepSize + s].1 )) (iota step)
                       let partresult = notSparseMatMult partmatrix voxels
                       in (output++partresult, run+1, runLen, stepSize, gridsize, entrypoints, totalLen)
               in (tail testmat)

     let forwardprojection_semiflat [r][a][n] (angles : [a]f32)
                         (rays : [r]f32)
                          (voxels : [n]f32)
                          (stepSize : i32) : []f32 =
               let gridsize = t32(f32.sqrt(r32((length voxels))))
               let halfsize = r32(gridsize)/2
               let entrypoints = convert2entry angles rays halfsize
               let totalLen = (length entrypoints)
               -- let runLen = if (totalLen/stepSize) == 0 then 1 else (totalLen/stepSize)
               let runLen = (totalLen/stepSize)
               let testmat = [0f32]
               let (testmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entrypoints, totalLen) = (testmat, 0, runLen, stepSize, gridsize, entrypoints, totalLen)
                   while ( run < runLen ) do
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       let intersections = map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints[(run*stepSize) : (run*stepSize + step)]
                       let shp = getshp intersections
                       let shp_scn = scan (+) 0 shp
                       let values_tmp = flatten(map(\r -> map(\(d,p)->(p,d))r)intersections)
                       let values = filter (\x -> x.1 != -1) values_tmp
                       let partresult = spMatVctMult values shp_scn voxels
                       in (output++partresult, run+1, runLen, stepSize, gridsize, entrypoints, totalLen)
               in (tail testmat)

     let forwardprojection_integrated [r][a][n] (angles : [a]f32)
                         (rays : [r]f32)
                          (voxels : [n]f32)
                          (stepSize : i32) : []f32 =
               let gridsize = t32(f32.sqrt(r32((length voxels))))
               let halfsize = gridsize/2
               let entryexitpoints =  convert2entryexit angles rays (r32(halfsize))
               let totalLen = (length entryexitpoints)
               -- let runLen = if (totalLen/stepSize) == 0 then 1 else (totalLen/stepSize)
               let runLen = (totalLen/stepSize)
               let testmat = [0f32]
               let (testmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entryexitpoints, totalLen) = (testmat, 0, runLen, stepSize, gridsize, entryexitpoints, totalLen)
                   while ( run < runLen ) do
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       let partresult = map(\(ent,ext) -> (reduce (+) 0 (flatten(map (\i ->
                                 calculate_fp_val ent ext i gridsize voxels
                            )((-halfsize)...(halfsize-1)))))) entryexitpoints[run*stepSize:run*stepSize+step]
                       in (output++partresult, run+1, runLen, stepSize, gridsize, entryexitpoints, totalLen)
               in (tail testmat)


     let backprojection_integrated  (angles : []f32)
                         (rays : []f32)
                         (projections : []f32)
                         (gridsize: i32)
                         (stepSize : i32) : []f32=
               let halfsize = gridsize/2
               let entryexitpoints =  convert2entryexit angles rays (r32(halfsize))
               let totalLen = (length entryexitpoints)
               let runLen = (totalLen/stepSize)
               -- result array
               let backmat = replicate (gridsize*gridsize) 0.0f32
               -- stripmined, sequential outer loop, mapped inner
               let (backmat, _, _, _, _, _, _) =
                   loop (output, run, runLen, stepSize, gridsize, entryexitpoints, totalLen) = (backmat, 0, runLen, stepSize, gridsize, entryexitpoints, totalLen)
                   while ( run < runLen ) do
                       -- if the number of entrypoints doesn't line perfectly up with the stepsize
                       let step = if (run+1)*stepSize >= totalLen then totalLen - run*stepSize else stepSize
                       let partmatresult = map(\j ->
                            (flatten(map (\i ->
                                calculate_bp_val (unsafe entryexitpoints[run*stepSize+j]).1 (unsafe entryexitpoints[run*stepSize+j]).2 i gridsize (unsafe projections[j])
                           )((-halfsize)...(halfsize-1))))) (iota step)
                       let transp = trans_map partmatresult gridsize
                       let partresult = map (\row -> reduce (+) 0 row) transp
                       -- add
                       let result = (map2 (+) partresult output)
                       in (result, run+1, runLen, stepSize, gridsize, entryexitpoints, totalLen)
               in backmat
}
