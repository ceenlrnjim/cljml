; TODO: adjust all these to be within a single top level package
(ns pca
  (:require [incanter.stats :as stats])
  (:require [incanter.core :as alg]))

; no x0 = 1 bias terms in these input matrices
; need mean normalization and feature scaling first
; TODO: move this to a utility package for use elsewhere in the library
(defn mean-normalize
  "Applies mean normalization to the specified vector if input examples (mxn)"
  [X]
  (let [Xprime (alg/trans X)
        means (map stats/mean Xprime)
        stddevs (map stats/sd Xprime)]
    (alg/trans ; convert back to X dimensions
      (alg/matrix
      ; Map over vectors of features ni for all m examples
        (map
          (fn [feature-vec mean-val stddev]
            (alg/div (alg/minus feature-vec mean-val) stddev))
          Xprime means stddevs)))))

; TODO: move this to a utility package for use elsewhere in the library
(defn feature-scale
  "Applies feature scaling using standardization to the supplied matrix of input examples."
  [X]
  ; TODO: can I optimize by combining with function above so I'm not recomputing means and standard deviations and reprocessing everything twice?
  ; TODO: is this required if we've done mean normalization?
  ; TODO: implement if so
  X)

(defn reduce-dimensions
  [X numdim]
  {:post [(= (alg/dim %) [(first (alg/dim X)) numdim])]}
  (let [[m n] (alg/dim X)
      ; Sigma = 1/m * (X' * X)
        Sigma (alg/mult (/ 1 m) (alg/mmult (alg/trans X) X))
      ; [U S V] = svd(Sigma);
        usv (alg/decomp-svd Sigma)
      ; Ureduce = U(:,1:k)
        Ureduce (alg/$ (range 0 numdim) (:U usv))
      ; z = X * Ureduce - assuming I have my matrix dimensions correct
        result (alg/mmult X Ureduce)]
    ;(println (map (fn [label mx] (str label ":" (alg/dim mx))) ["X" "Sigma" "U" "Ureduce" "result"] [X Sigma (:U usv) Ureduce result]))
    ;(println X)
    ;(println result)
    result))

(defn retained-variance
  [X Xapprox]
  ; TODO: how do I get the norm for vectors of different lengths?  Or what am I forgetting about xapprox's values?
  ; 1/m*(sum over m ||xi - xapproxi||^2 / (1/m*(sum over ||xi||^2))
  nil)

; probably want to include functions for measuring retained variance (avg squared projection error / total variation) =

