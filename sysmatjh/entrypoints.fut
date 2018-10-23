
import "intersect_lib"
open Intersections

entry main
  (angles:  []f32)
  (rays:    []f32)
  (gridsize: i32)
    : ([]f32,[]f32)=
      let halfsize = r32(gridsize)/2
      in unzip(map(\(p,cs) -> p) (convert2entry angles rays halfsize))
