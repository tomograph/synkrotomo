import "intersect_lib"
open Intersections

module Projection = {

  --segmented scan with (+) on floats:
  let sgmSumF32 [n]
                (flg : [n]i32)
                (arr : [n]f32) : [n]f32 =
    let flgs_vals = scan ( \ (f1, x1) (f2,x2) -> let f = f1 | f2 in if f2 > 0 then (f, x2) else (f, x1 + x2) ) (0,0.0f32) (zip flg arr)
    let (_, vals) = unzip flgs_vals
    in vals

  -- segmented scan with (+) on ints:
  let sgmSumI32 [n]
                (flg : [n]i32)
                (arr : [n]i32) : [n]i32 =
    let flgs_vals = scan ( \ (i1, x1) (i2,x2) -> let i = i1 | i2 in if i2 > 0 then (i, x2) else (i, x1 + x2) ) (0,0i32) (zip flg arr)
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


  let notSparseMatMult [num_rows] [num_cols]
                        (mat_vals : [num_rows][num_cols](f32,i32))
                        (vect : []f32) : [num_rows]f32 =
    map (\row -> reduce (+) 0 <| map (\(v, ind) -> unsafe (if ind == -1 then 0.0 else v*vect[ind]) ) row ) mat_vals

  -- sparse matrix vector multiplication
  let spMatVctMult [num_elms] [vct_len] [num_rows]
                    (mat_val : [num_elms](i32,f32))
                    (shp_scn : [num_rows]i32)
                    (vct : [vct_len]f32) : [num_rows]f32 =
    let len = shp_scn[num_rows-1]
    let shp_inds = map (\i -> if i==0 then 0 else unsafe shp_scn[i-1] ) (iota num_rows)
    let flags = scatter ( replicate len 0) shp_inds ( replicate num_rows 1)
    let prods = map (\(i,x) -> x*(unsafe vct[i])) mat_val
    let sums = sgmSumF32 flags prods
    let mat_inds = map (\i -> i-1) shp_scn
    in map (\i -> unsafe sums[i]) mat_inds

    -- gets the shape of a matrix - i.e number of entries pr. row when mat is in the format [[(d,i),(d,i)...]] where d is data
    -- and i is column index and -1 means no data
  let getshp (matrix: [][](f32,i32)) : []i32 =
    let dp = unzip_d2 matrix
    let flagsforindexes = map(\r -> map(\p -> if p == -1 then 0 else 1)r)dp.2
    in map(\f -> reduce (+) 0 f)flagsforindexes

    -- let transMat [num_rows] [num_cols] (mat : [num_rows][num_cols](f32,i32)) =
    -- map (\c ->  map (\r -> mat[r][c] ) mat ) (iota num_cols)
    -- helper function to determine if we entered new segment of consecutive numbers in an array.
  let isnewsegment (i: i32) (arr: []i32) : bool =
    i!=0 && (unsafe arr[i]) != (unsafe arr[i-1])

  let backprojection  (rays : []f32)
                      (angles : []f32)
                      (projections : []f32)
                      (gridsize: i32) : []f32=
    let halfsize = r32(gridsize)/2
    let entrypoints = convert2entry angles rays halfsize
    let intersections = map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints
    --convert to triples (data,ray,pixel)
    let triples_tmp = flatten(map(\i -> map(\v -> (v.1, i, v.2))(unsafe intersections[i])) (iota (length intersections)))
    -- remove none values
    let triples = filter (\x -> x.3 != -1) triples_tmp
    -- sort by pixel indexes
    let pixelsorted = rsort triples
    -- slplit int three arrays in order to use pixels only for shp
    let (data,rays,pixels) = unzip3 pixelsorted
    let num_pixels = length pixels
    -- contains sum of values where a row ends since columns will be rows.
    let shp_scn_tmp = map (\i -> if (i == num_pixels || (isnewsegment i pixels)) then i else 0) (iota (num_pixels+1))
    let shp_scn = filter (\p -> p != 0) shp_scn_tmp
    let values = map(\x-> (x.2,x.1)) pixelsorted
    in spMatVctMult values shp_scn projections

  -- let bpMult [num_rows] [num_cols] (mat : [num_rows][num_cols](f32,i32)) (vect : []f32) =
  --   -- let (vals, inds) = unzip_d2 mat
  --   let zm = map (\row -> map (\(v, ind) -> if ind == -1 then (0.0, ind) else (v, ind) ) row ) mat
  --   in map (\r -> map (\i -> zm.1[r][i] * vect[i]) (iota num_cols) ) (iota num_rows)

  -- let backprojectionNested  (rays : []f32)
  --                           (angles : []f32)
  --                           (projections : []f32)
  --                           (gridsize: i32) =
  --   let halfsize = r32(gridsize)/2
  --   let entrypoints = convert2entry angles rays halfsize
  --   let intersections = map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints
  --   let treated = (transpose (map (\row -> map (\(v, ind) -> if ind == -1 then 0.0 else v ) row ) intersections))
  --   in bpMult treated projections
    -- in map (\row -> reduce (+) 0 <| map (\i -> unsafe (treated[row][i] * projections[i])) (iota ((length rays) * (length angles))) ) (iota (gridsize*gridsize))
    -- in map (\row -> reduce (+) 0 <| map (\(v, ind) -> unsafe (if ind == -1 then 0.0 else v*projections[i]) ) row ) (iota num_cols)
    -- in notSparseMatMult AT projections --set output size
    -- in map (\row ->
    --   reduce (+) 0 <| map (\(v, ind) -> Lambda-function) row
    -- ) AT

  let forwardprojection (rays : []f32)
                        (angles : []f32)
                        (voxels : []f32) =
    let gridsize = t32(f32.sqrt(r32((length voxels))))
    let halfsize = r32(gridsize)/2
    let entrypoints = convert2entry angles rays halfsize
    let intersections = map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints
    -- let shp = getshp intersections
    -- let shp_scn = scan (+) 0 shp
    -- let values_tmp = flatten(map(\r -> map(\(d,p)->(p,d))r)intersections)
    -- let values = filter (\x -> x.1 != -1) values_tmp
    -- in spMatVctMult values shp_scn voxels
    in notSparseMatMult intersections voxels
    -- in map (\(p,sc) -> (lengths gridsize sc.1 sc.2 p)) entrypoints

  }
