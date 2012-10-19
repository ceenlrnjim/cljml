(ns kmeans
  (:require [incanter.core :as alg]))

; TODO: excellent opportunity for the reducers
(defn norm
  "Returns the norm of the specified sequence - usually a vector"
  [s]
  (Math/sqrt
    (reduce #(+ %1 (Math/pow %2 2)) 0 s)))

(defn initialize-centroids
  "Returns K centroid locations (Kxn) matrix, given training set s (mxn matrix)"
  [K trainset]
  {:post [(let [[tsrows tscols] (alg/dim trainset)] (= (alg/dim %) [K tscols]))] }
  (alg/matrix (take K (shuffle (alg/to-list trainset)))))

(defn closest-pt
  "Returns the index in cseq of the c that is closest to point example xi.
  xi is expected to be 1xn (slice of total data matrix)
  cseq is expected to be Kxn"
  [cseq xi]
  {:pre [(let [[xrows xcols] (alg/dim xi) [crows ccols] (alg/dim cseq)] (and (= xcols ccols)))] }
                  ; sequence of [cix distance]
  (let [distances (map-indexed #(vector %1 (norm (alg/pow (alg/minus xi %2) 2))) cseq)]
    (first ; [index cost] - want just the index
      (reduce
        (fn [best i] (if (> (best 1) (i 1)) i best))
        distances))))

(defn capture-target
  "returns a map from centroid (nx1) to collection of training set points (sxn) where s is the number of points captured"
  [trainset c]
    ; end up with map from cix -> [x1 x4 x6....]
    (into {}
      ; convert lists from group-by into sets
      (for [[k v]
        ; group all points that are closest to the same centroid
        (group-by (partial closest-pt c) trainset)]
        [k (set v)])))

(defn move-centroids
  "Returns the new value of c for the specified map from centroid index to list of captured points"
  [mapping]
  ; convert from map [ix -> set of locations] to vector of new location values for c
  (alg/matrix
    ; TODO: vectorize this implementation
    (map
      (fn [[ix pts]]
        (alg/div (reduce alg/plus pts) (count pts)))
      (sort mapping)))) ; need value to be at the index that was the key in the ampping

; TODO: make this an optional parameter to kmeans, defaulting to unbound
(def MAX_ITERS 100)

(defn kmeans
  "Groups the specified set of n-dimensional locations (mxn matrix where m is the number of examples
  in the training set and n is the number of features) into the specified number of clusters.
  Returns a map of index to a set of matrices representing the points in the cluster"
  [X clusters]
  (let [initc (initialize-centroids clusters X)]
    ; iterate until values don't change
    (loop [c initc
           lastcap {}
           iters 0]
      (let [cap (capture-target X c)]
        (if (or (= cap lastcap) (> iters MAX_ITERS)) lastcap
          (recur (move-centroids cap) cap (inc iters)))))))

