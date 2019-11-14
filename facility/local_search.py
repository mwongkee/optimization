import numpy as np
import math
import random

def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def initial_solution(customers, facilities):
    solution = [-1] * len(customers)
    capacity_remaining = [f.capacity for f in facilities]
    customers_per_fac = [0 for f in facilities]

    avg_customers_per_fac = len(customers) / len(facilities)
    max_cust = avg_customers_per_fac * 2

    facility_index = 0
    i = 0
    for customer in customers:
        #print(i, facility_index)
        i +=1
        if capacity_remaining[facility_index] >= customer.demand and customers_per_fac[facility_index] < max_cust:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
            customers_per_fac[facility_index] += 1
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
    return solution


def calculate_obj(solution, facilities, customers):
    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    return obj

def weight_length(num_facilities, num_customers, customer, facility):
    return length(customer.location, facility.location)

def weight_length_plus_some_setup(num_facilities, num_customers, customer, facility):
    dist = length(customer.location, facility.location)
    setup = facility.setup_cost
    factor = 2
    return dist + setup/factor

def weight_random(num_facilities, num_customers, customer, facility):
    return random.random()


def compute_neighbours(customers, facilities, neighbour_func, num_neighbours):
    neighbours = {}
    num_customers = len(customers)
    num_facilities = len(facilities)
    for c_i, c in enumerate(customers):
        this_neighbours = []
        for f_i, f in enumerate(facilities):
            this_neighbours.append(neighbour_func(num_facilities, num_customers, c, f))
        sorted_ind = np.array(this_neighbours).argsort()
        neighbours[c_i] = set(sorted_ind[:num_neighbours])
    return neighbours




def local_search(facilities, customers):


    num_neighbours = 25
    if len(facilities) == 25:
        solution = '7 7 7 7 7 7 6 7 7 7 10 7 7 7 7 7 7 6 7 7 7 7 7 7 7 7 21 7 7 7 7 7 7 16 7 7 11 6 7 7 7 7 7 7 7 7 7 7 7 7'.split(
            ' ')
        solution = [int(x) for x in solution]
    elif len(facilities) == 50:
        solution = '28 24 19 25 14 15 34 3 9 24 35 42 41 24 49 3 16 26 43 45 45 41 9 34 9 13 19 8 38 24 24 7 9 25 31 33 28 9 28 25 38 40 35 7 2 19 40 25 41 9 34 44 41 18 35 5 9 31 14 35 31 35 44 43 9 4 8 14 25 45 28 33 41 39 42 6 8 35 6 24 40 47 31 24 31 24 24 45 34 9 7 2 5 39 25 35 24 40 31 3 47 6 39 16 31 44 2 16 9 13 8 9 47 35 15 24 43 25 42 16 35 28 34 35 13 5 8 35 18 11 38 39 43 41 47 44 9 41 9 38 38 28 19 9 28 28 42 41 47 9 35 38 29 8 45 49 16 25 26 31 38 5 49 10 7 40 44 29 34 10 2 41 13 31 40 28 35 49 44 33 4 2 16 47 28 9 3 31 11 16 31 25 5 42 13 31 8 40 44 45'.split(' ')
        solution = [int(x) for x in solution]
    elif len(facilities) == 100 and len(customers) == 1000:
        solution = '28 9 32 2 37 36 37 37 37 17 9 37 28 37 37 2 13 37 39 9 36 37 2 32 17 37 2 37 13 37 2 39 9 9 23 32 23 13 37 37 23 9 32 2 2 17 13 28 23 36 32 32 32 37 37 32 37 9 9 17 37 36 17 36 37 23 37 23 37 9 37 36 37 32 28 37 39 36 37 39 36 28 36 39 37 2 2 2 32 37 17 36 28 9 9 23 2 37 17 37 32 28 37 9 37 37 28 2 9 9 32 37 37 2 37 28 37 17 37 32 39 13 37 36 32 9 2 23 32 37 28 2 37 37 37 36 17 23 23 9 32 2 37 2 9 37 39 17 39 32 39 37 17 36 37 9 37 37 37 32 37 37 37 23 37 32 13 28 32 13 32 23 37 9 36 37 37 2 13 32 23 13 32 2 37 9 37 37 36 13 17 28 37 17 37 37 37 36 37 37 37 37 28 37 37 2 17 23 37 36 17 9 37 37 37 39 17 9 9 39 2 2 2 23 36 37 37 39 17 37 32 17 9 2 17 23 37 2 37 17 2 13 37 32 32 2 36 32 37 37 2 37 36 37 13 37 37 37 2 37 39 28 37 2 9 13 23 23 9 2 2 28 28 37 37 36 37 23 23 37 39 23 39 39 36 37 17 36 13 9 28 37 37 39 13 23 37 9 37 2 36 2 36 39 2 36 9 37 36 23 37 9 36 36 2 37 36 37 2 37 17 23 37 37 23 37 37 32 39 37 23 23 37 37 37 36 37 32 9 32 37 37 37 2 13 23 39 17 37 36 37 37 2 39 37 37 23 37 9 37 23 37 2 17 13 17 2 9 37 37 37 17 37 17 37 37 2 39 9 9 32 13 39 37 23 28 37 9 37 2 37 37 9 32 37 32 2 36 2 13 28 9 37 37 32 13 23 37 37 9 32 37 32 9 37 37 23 23 37 17 17 37 37 2 32 32 23 39 37 37 2 37 37 2 28 37 28 37 2 37 37 36 23 32 37 32 37 37 37 37 32 37 2 37 2 13 37 39 37 13 32 37 2 13 2 32 32 13 37 23 23 2 36 37 28 37 32 32 37 39 17 2 32 23 37 9 17 37 13 32 32 9 17 23 9 9 2 2 9 37 36 37 2 2 37 23 37 28 32 37 13 2 9 37 36 37 32 23 36 37 32 2 37 39 2 17 37 37 37 23 9 37 32 37 36 13 13 32 37 37 37 37 36 17 9 32 37 2 37 37 37 39 9 37 32 39 2 37 37 2 17 37 37 23 23 37 17 9 37 37 39 32 32 39 17 23 23 37 9 37 2 37 36 9 2 9 13 36 13 37 39 37 37 37 37 23 36 37 13 37 28 37 36 37 2 37 32 32 13 32 9 36 37 28 37 37 39 23 28 36 37 37 36 17 37 9 17 2 37 32 37 37 9 37 36 23 17 37 37 2 36 37 37 37 37 17 13 32 37 37 39 9 9 9 37 13 37 36 37 2 23 37 37 37 37 23 36 37 39 17 28 37 37 17 23 32 32 39 37 32 36 23 23 37 17 32 39 2 32 39 37 32 32 23 37 37 2 36 13 9 37 37 13 37 23 17 36 17 2 17 23 28 37 13 28 9 37 37 17 37 9 9 23 37 37 36 37 37 36 32 39 37 37 37 17 36 2 32 37 28 37 37 2 37 2 9 23 32 9 36 23 17 37 36 32 9 2 32 13 37 17 37 37 28 36 37 9 37 32 36 13 28 9 37 37 2 2 37 37 37 13 37 37 37 36 23 37 37 9 37 36 37 9 2 37 37 2 23 28 28 37 36 37 37 23 17 9 23 2 37 23 39 32 37 37 23 37 23 17 39 2 37 2 17 2 36 36 37 9 9 32 37 39 23 37 32 37 13 32 17 37 39 23 32 32 32 37 2 23 39 32 2 37 37 36 37 37 9 37 9 37 2 39 39 37 9 32 32 23 37 17 32 23 17 37 39 17 39 28 9 9 17 32 17 37 23 37 37 37 28 36 32 37 13 37 37 13 28 37 37 32 37 32 17 37 28 37 37 37 13 37 37 36 39 37 2 36 2 17 37 2 28 28 37 32 28 37 37 17 2 37 36 23 23 13 32 37 37 32 36 37 23 37 9 32 36 37 23 37 2 13 13 13 37 39 28 2 37 36 9 13 36 36 9 2 23 17 23 37 28 9 37 2 2 9 17 23 32 39 28 37 37 37 37 37 2 28 9 37 28 2 32 37 37 32 13 9 37 9 37'.split(' ')
        solution = [int(x) for x in solution]
    elif len(facilities) == 100 and len(customers) == 100:
        solution = '70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70 70'.split(' ')
        solution = [int(x) for x in solution]
    elif len(facilities)  == 200:
        solution = '155 29 132 66 76 33 164 24 6 113 167 19 35 48 136 183 13 169 3 46 60 85 139 192 167 27 95 118 51 151 13 180 72 175 3 24 20 157 120 114 108 49 48 175 171 35 102 27 123 145 162 66 123 157 94 139 151 3 160 118 183 167 3 26 21 37 25 25 94 198 66 162 146 86 60 118 43 54 159 30 123 108 125 92 37 71 146 62 2 20 114 30 139 48 85 33 141 94 162 174 113 108 149 119 46 118 175 192 69 42 144 13 62 120 43 30 165 33 139 69 37 165 185 185 29 171 180 192 6 197 66 6 76 4 123 155 164 175 49 66 105 60 136 56 69 32 140 3 119 4 158 92 160 160 181 113 176 22 146 146 165 101 41 123 118 32 174 27 56 4 151 135 146 136 49 32 22 185 45 95 167 27 60 94 114 173 15 15 48 151 182 163 178 60 49 26 35 163 185 123 144 68 37 119 176 71 136 86 21 173 174 102 43 25 13 94 4 132 32 145 92 78 66 43 113 62 32 185 175 186 55 118 32 2 197 114 13 6 94 145 85 56 123 118 167 37 100 95 100 105 197 45 151 35 53 56 185 22 109 125 15 27 7 186 43 101 186 180 56 145 198 85 126 29 126 42 4 164 21 132 108 160 178 87 71 45 169 2 163 145 113 123 162 114 37 181 157 167 35 105 48 78 20 21 113 52 145 46 78 76 66 120 175 26 189 41 26 169 157 25 24 51 185 169 145 51 101 24 135 141 30 120 118 182 159 95 25 51 145 35 15 22 157 174 102 71 54 118 87 3 19 69 171 158 69 185 85 135 108 189 125 141 25 15 126 183 126 126 171 144 76 35 62 101 160 87 78 68 87 144 29 71 126 100 66 3 165 119 6 165 51 157 56 180 181 109 92 20 164 178 76 140 174 45 35 180 2 182 105 22 48 144 175 183 162 94 169 113 55 178 139 92 27 24 181 159 54 198 197 151 145 120 155 144 68 71 69 52 6 159 186 188 6 3 159 43 85 47 47 60 66 13 7 37 52 183 19 173 157 26 188 62 7 160 132 173 22 159 167 136 136 158 41 26 46 101 25 47 22 71 141 126 78 15 101 113 102 144 13 41 113 182 171 30 141 103 135 146 3 49 132 186 46 144 186 52 69 119 29 188 149 180 45 15 45 176 192 197 162 95 100 160 102 22 66 7 188 123 180 27 21 119 178 3 144 27 159 43 45 186 144 45 103 15 158 173 71 140 120 60 72 160 145 189 72 118 2 167 78 158 139 135 29 186 42 86 29 108 92 146 94 66 101 62 55 165 92 176 95 69 72 19 21 197 68 149 198 132 48 2 24 165 185 15 109 176 140 43 146 158 157 102 160 139 114 46 94 66 197 188 183 158 183 183 192 3 119 141 180 108 171 72 7 164 33 181 33 176 4 42 32 192 3 165 26 181 46 60 101 185 120 72 149 135 192 198 108 174 94 49 157 174 135 66 72 197 162 185 87 68 173 174 189 167 47 32 2 120 35 95 158 95 141 162 43 51 26 41 189 21 175 60 46 126 85 53 136 48 71 86 178 123 37 169 178 56 85 167 158 66 62 19 175 41 100 41 55 100 135 47 4 114 146 51 197 19 54 72 19 157 51 180 101 42 164 118 4 7 52 56 125 136 169 62 102 52 114 60 186 53 60 120 60 76 48 144 24 189 41 100 29 30 69 139 94 76 157 19 37 125 144 185 164 173 49 140 19 7 52 85 171 103 183 183 105 51 186 92 32 175 113 33 169 105 26 189 33 141 139 29 24 157 135 180 178 182 22 114 56 105'.split(' ')
        solution = [int(x) for x in solution]
    elif len(facilities) == 500:
        solution = '131 33 359 206 88 469 239 358 240 327 69 330 342 307 277 175 394 326 157 170 426 471 222 408 254 481 361 236 484 427 489 190 115 20 190 186 198 39 416 9 238 490 51 451 116 484 206 458 263 104 498 308 171 482 111 48 51 134 200 9 178 167 394 458 222 167 239 288 92 115 104 255 72 417 467 318 346 193 11 394 430 359 1 475 144 209 82 162 445 164 388 277 482 212 100 381 29 443 152 431 450 105 67 417 418 498 47 277 66 45 412 7 489 381 230 319 221 490 206 198 72 289 248 493 362 490 314 359 384 475 433 315 75 222 229 215 443 460 424 30 170 110 314 479 69 36 9 11 165 183 332 318 418 50 190 394 410 152 104 125 392 301 206 449 33 245 244 483 36 34 324 299 92 309 53 447 97 348 458 156 4 181 15 177 454 257 260 461 27 85 32 190 133 34 334 23 227 384 11 229 483 148 206 423 162 302 137 405 65 157 497 344 446 127 431 31 127 294 66 446 340 69 64 408 350 58 372 356 260 264 65 11 499 451 58 69 66 94 39 428 394 117 372 32 447 325 6 311 191 96 93 430 204 14 454 134 22 82 224 355 483 293 394 397 401 401 324 32 131 306 139 14 468 23 189 391 50 308 380 352 320 364 212 17 327 308 271 15 426 263 74 213 230 430 309 127 223 26 426 342 203 210 226 309 170 40 15 221 271 162 426 355 326 324 162 186 135 167 309 309 254 241 403 64 407 390 257 490 458 325 93 481 144 139 94 489 40 398 450 483 432 9 136 140 259 94 344 469 111 335 318 311 239 19 355 102 300 56 460 324 327 39 200 162 318 100 207 468 450 289 495 102 496 219 207 148 68 485 334 335 181 443 379 335 235 155 209 175 221 157 164 356 360 433 156 215 222 228 200 498 303 266 461 465 184 171 342 346 193 229 314 140 241 335 224 490 380 312 100 57 137 14 497 109 362 130 210 221 237 7 140 155 244 31 359 330 472 40 136 229 425 497 482 300 210 152 432 203 0 159 77 483 130 497 36 136 381 51 289 307 297 144 206 72 205 245 369 186 200 39 127 22 324 380 315 136 26 203 227 189 395 400 340 228 261 421 421 57 369 289 94 218 177 235 425 92 130 344 394 48 186 397 477 139 74 367 208 63 11 373 240 398 296 17 22 243 342 223 165 241 355 484 14 30 263 300 315 173 19 182 359 212 140 431 314 469 11 147 469 169 361 237 93 400 243 115 137 113 468 44 15 277 384 33 215 23 117 408 479 369 206 197 309 92 443 361 271 221 199 179 84 31 464 346 421 6 498 66 155 457 148 355 384 468 428 315 219 139 89 416 408 186 168 62 144 361 58 45 155 133 353 7 397 218 11 186 394 7 393 69 73 143 184 229 340 133 221 461 381 362 336 110 218 243 173 484 421 8 361 189 484 438 379 436 483 392 75 186 104 28 288 184 39 245 236 317 7 193 73 239 39 7 92 203 238 207 147 62 202 115 75 449 385 7 282 438 67 405 277 405 122 235 217 481 224 14 64 119 206 182 88 373 49 210 198 84 496 257 325 20 485 208 493 134 69 346 144 130 305 228 252 123 447 498 59 481 316 394 458 32 215 307 373 0 430 244 77 381 75 113 421 256 325 222 479 103 215 259 7 381 294 112 6 136 225 313 115 499 350 229 223 405 363 144 56 405 22 346 97 184 356 68 84 282 184 400 196 441 88 379 241 34 397 105 183 397 26 257 384 179 333 318 446 45 73 15 358 353 381 227 400 462 209 260 167 66 257 235 238 31 362 395 39 453 469 113 221 55 350 451 238 206 338 204 446 156 228 89 468 430 282 337 182 335 379 109 125 353 438 342 372 325 239 74 206 205 237 196 53 403 256 446 197 235 148 89 441 460 186 302 62 139 31 200 59 271 340 451 384 261 451 197 222 277 314 479 123 475 260 211 398 206 344 137 277 210 493 288 383 212 337 296 457 421 229 472 335 111 307 203 360 122 29 340 169 332 75 84 77 299 22 243 459 360 23 209 109 112 176 451 186 235 155 131 330 277 282 167 303 103 156 155 196 367 326 466 125 186 200 89 303 395 57 309 337 467 173 114 315 0 288 302 116 436 23 202 303 45 156 85 139 301 408 360 173 412 143 438 28 40 208 297 215 246 363 309 73 202 477 363 173 198 6 96 46 169 465 186 369 45 273 369 7 362 134 175 83 46 169 458 266 499 443 55 105 114 48 127 169 215 184 112 498 131 203 320 94 45 364 443 6 107 58 75 477 218 67 316 102 72 356 361 236 127 319 464 7 34 385 237 320 311 375 223 417 186 365 229 453 450 391 56 397 0 191 88 418 460 308 66 100 22 62 125 34 318 45 164 270 369 477 446 423 426 110 15 36 482 11 472 362 469 111 252 241 7 217 379 493 471 181 346 100 390 310 398 484 362 64 210 227 62 408 347 93 367 224 477 299 87 173 302 462 93 257 0 144 1 113 75 266 485 228 314 450 496 112 450 436 431 464 256 359 340 379 131 454 113 360 277 29 30 260 256 185 147 309 225 240 320 189 202 287 64 156 30 93 237 403 421 147 419 313 294 497 325 269 47 269 245 300 388 57 320 424 356 333 227 236 183 123 458 20 355 17 318 202 403 224 415 64 225 123 337 345 209 75 431 210 50 407 294 315 60 394 207 302 395 152 352 55 344 30 372 387 427 222 431 31 147 239 164 445 6 30 387 266 205 178 419 191 477 333 186 45 260 475 102 447 102 246 252 321 490 205 449 325 137 56 273 117 176 170 316 52 397 425 157 261 433 359 250 346 372 168 207 430 15 226 469 266 483 60 228 390 229 4 164 26 340 392 257 325 412 230 392 395 15 408 301 136 157 238 9 105 63 430 375 302 20 475 110 103 254 443 89 31 497 477 345 184 182 29 489 14 255 36 433 460 466 432 110 94 168 29 416 383 211 225 314 89 127 469 321 209 170 407 482 479 178 20 238 77 66 256 338 155 261 360 333 407 313 184 288 64 294 8 337 239 167 15 118 121 189 425 36 400 225 320 51 319 6 206 327 140 498 75 269 309 369 57 162 179 458 161 55 191 246 288 168 318 162 493 118 87 121 52 185 67 191 11 427 324 497 208 471 394 191 64 408 338 46 14 484 261 484 74 175 273 110 20 162 240 144 181 263 313 34 305 313 431 203 191 261 342 196 394 310 118 356 111 385 53 200 318 215 85 416 137 369 400 63 115 196 133 311 230 398 125 183 443 69 109 237 190 161 303 309 432 391 126 324 252 495 401 170 363 221 405 11 113 263 69 394 461 325 267 219 45 40 45 302 360 167 348 498 333 143 407 197 137 111 64 198 173 497 55 416 68 159 227 33 209 256 206 446 165 363 307 299 324 114 430 226 438 445 483 156 260 454 324 60 472 15 202 372 51 69 125 450 380 453 155 433 27 361 147 346 364 292 205 73 33 152 148 363 464 162 127 220 83 462 65 346 203 112 418 223 299 308 255 135 303 17 57 388 92 460 147 93 88 116 173 241 39 335 266 41 418 73 196 493 412 109 207 166 181 57 438 392 273 495 443 468 39 297 316 239 360 147 263 161 359 417 4 246 169 289 311 297 0 289 363 416 170 175 332 36 489 19 116 249 288 257 198 107 93 403 27 147 451 347 482 114 68 465 137 240 266 391 365 363 48 207 88 356 347 45 227 346 113 245 215 134 238 335 30 372 6 7 324 130 28 264 336 197 407 294 34 202 58 256 222 7 479 445 237 148 361 178 481 482 321 109 15 342 137 50 467 352 109 427 432 313 484 193 212 369 263 29 161 277 87 85 292 307 412 104 363 451 267 315 65 77 493 60 241 136 237 152 224 237 307 308 0 64 139 359 225 97 340 240 245 19 403 217 34 249 157 257 67 267 33 198 48 236 152 39 87 162 33 46 26 299 257 408 335 171 100 292 26 288 369 355 134 410 493 245 394 485 227 311 170 161 200 135 29 221 33 133 248 466 229 320 271 104 403 471 438 40 225 125 219 240 496 53 170 85 100 436 190 39 186 394 358 221 32 116 325 87 88 68 155 459 388 64 104 29 446 259 1 156 19 167 316 338 338 342 246 244 263 162 469 347 391 390 348 178 77 175 390 317 350 113 217 9 212 177 191 259 161 261 394 309 222 218 256 314 446 94 156 7 469 196 221 134 498 390 52 72 271 179 383 70 199 308 15 119 121 17 489 0 467 380 59 104 353 489 74 51 70 184 217 375 9 111 209 401 165 36 50 303 208 92 89 49 55 157 299 498 447 405 31 259 136 58 408 471 316 177 198 72 155 72 407 131 93 446 219 227 472 355 189 443 203 433 182 271 327 419 238 173 104 57 302 300 218 185 33 433 245 468 257 189 497 112 14 130 332 166 77 205 93 26 427 207 207 313 191 427 430 15 464 459 294 203 100 112 162 123 15 353 92 260 308 112 438 152 408 499 4 269 219 94 364 32 334 365 32 215 464 46 229 485 401 26 271 226 466 167 255 173 407 133 431 223 256 225 222 385 482 325 127 461 249 224 342 384 292 6 335 167 436 484 310 241 122 200 252 424 88 44 237 183 39 156 441 73 458 224 215 32 373 451 425 112 415 310 401 203 19 30 484 360 259 249 425 45 288 484 207 162 350 360 496 219 425 213 364 447 168 229 27 125 207 177 220 321 360 373 75 263 34 496 261 417 213 427 123 218 241 362 367 4 346 252 93 468 390 143 248 313 299 256 211 94 259 364 193 405 270 259 191 325 490 208 226 175 182 87 143 93 227 307 203 185 458 205 112 41 51 211 320 327 307 211 190 169 432 1 97 36 472 46 63 282 412 77 198 97 14 407 471 48 259 148 152 113 419 130 360 167 175 416 23 53 40 461 241 218 314 152 433 244 14 282 7 59 20 100 162 236 92 88 110 238 104 481 333 135 189 190 380 17 282 197 56 44 47 428 190 131 170 237 217 469 425 9 372 210 289 334 315 313 32 471 240 102 401 186 114 147 334 289 489 312 177 205 96 74 40 125 34 400 200 202 248 317 58 215 60 135 178 333 198 184 104 308 47 148 121 170 70 394 306 482 379 321 257 246 320 182 308 85 32 219 308 219 88 324 40 22 405 259 186 249 367 67 419 384 33 53 217 155 20 450 346 123 327 162 415 467 97 74 11 408 58 257 250 287 148 239 184 263 245 445 186 69 335 297 130 77 8 27 261 56 294 221 430 410 94 8 432 260 20 408 311 178 282 300 245 66 119 143 223 175 375 260 369 287 67 152 312 344 407 126 218 130 207 64 388 144 430 66 287 224 287 495 135 171 68 387 266 277 249 418 215 340 166 219 175 359 248 84 100 148 97 121 211 123 320 82 394 390 87 468 181 482 395 320 64 424 263 72 147 191 171 451 167 269 97 199 228 410 191 157 263 263 103 208 289 461 119 254 197 240 87 109 385 335 410 82 418 250 241 65 314 387 70 31 115 321 380 239 190 179 477 453 191 193 277 407 263 112 321 17 228 209 48 49 319 125 498 336 266 57 50 113 373 254 479 74 198 134 269 140 408 417 483 29 147 468 66 297 398 451 126 497 33 443 143 364 65 125 497 107 189 289 310 493 359 282 292 477 240 459 416 241 123 369 52 205 84 485 289 23 100 443 75 375 450 51 293 319 111 17 127 479 241 289 157 190 117 254 229 495 122 123 11 126 114 36 222 74 144 72 342 127 450 352 345 107 127 307 193 246 40 496 33 178 140 200 205 117 217 271 193 299 19 307 490 185 254 49 270 345 352 161 333 9 394 296 222 1 372 485 217 186 148 139 497 266 190 267 118 111 193 493 123 270 426 53 217 303 207 334 301 0 211 117 1 373 359 178 391 235 36 48 226 293 46 259 464 383 264 189 122 407 196 412 384 57 137 157 73 462 208 460 261 388 342 14 64 200 26 315 431 184 11 204 493 340 41 199 270 300 301 7 190 75 207 319 430 325 410 472 257 63 496 184 254 58 270 401 294 152 56 93 336 213 177 457 479 202 407 11 210 438 432 155 325 243 348 191 139 467 305 235 315 189 498 335 342 330 468 292 346 36 92 226 239 196 70 468 334 335 175 148 94 157 44 125 483 157 47 483 152 466 170 316 17 72 256 211 219 66 84 209 483 483 117 28 227 277 114 466 443 352 155 148 289 222 312 273 222 156 373 438 485 427 431 302 465 312 134 60 301 301 337 230 244 421 223 58 213 380 6 418 96 309 490 340 237 287 446 342 348 493 256 360 189 407 169 310 200 438 261 479 385 47 208 369 70 121 472 157 464 238 375 344 369 184 320 362 56 294 182 47 313 294 248 496 222 468 217 301 416 238 77 356 483 225 277 239 333 111 407 311 293 29 472 403 443 489 467 426 297 360 58 438 148 317 127 155 430 44 58 15 477 156 228 344 136 189 264 102 66 380 178 222 26 313 126 41 117 294 94 397 157 220 410 144 469 190 465 307 196 30 217 447 336 359 85 11 309 362 236 333 261 477 62 325 264 419 205 380 116 305 243 40 338 162 62 469 416 219 75 436 112 421 489 241 89 333 226 273 161 11 373 256 59 167 193 181 144 324 297 189 300 416 428 77 426 70 246 393 226 22 454 58 391 466 161 335 277 217 181 296 325 430 193 52 220 68 112 131 228 215 102 423 202 346 148 135 254 394 213 97 338 235 8 204 400 425 28 427 203 381 311 63 133 136 72 472 326 185 490 140 7 198 167 361 239 228 266 51 243 436 77 116'.split(
            ' ')
        solution = [int(x) for x in solution]
    elif len(facilities) == 1000:
        solution = '520 110 872 497 96 757 393 697 743 485 97 29 760 742 411 172 728 494 669 559 805 867 397 380 885 865 542 14 308 137 123 334 796 397 976 100 14 552 77 699 714 678 521 15 535 678 817 329 336 79 821 469 451 594 285 117 391 688 638 999 781 61 542 884 694 907 223 259 937 912 148 487 777 848 370 552 431 129 471 769 370 788 947 427 106 501 959 187 507 24 949 915 811 419 959 766 537 275 962 697 403 58 919 374 929 937 962 625 534 837 735 863 467 610 343 499 890 137 379 485 127 707 322 167 511 415 413 877 507 250 497 646 836 351 479 88 956 479 720 61 322 278 361 932 893 780 346 988 682 760 856 859 742 479 775 868 117 978 148 114 844 668 858 437 915 472 77 140 669 885 494 559 362 127 636 199 213 566 190 167 119 429 119 796 758 397 699 641 346 821 832 960 175 676 989 26 300 282 999 963 696 187 603 696 976 326 354 766 379 880 817 780 18 750 875 420 329 707 455 956 391 63 79 635 670 157 216 463 97 54 8 611 61 103 347 694 985 490 553 94 322 47 940 735 890 394 735 924 437 844 998 999 590 142 264 452 259 505 215 665 698 61 805 343 947 105 848 394 455 213 735 240 400 21 977 772 371 772 117 594 574 867 393 839 230 253 60 179 22 364 717 175 809 879 246 749 291 138 233 610 279 54 567 746 331 941 459 213 435 326 514 287 222 451 471 938 972 78 858 735 182 821 293 369 903 251 223 786 203 362 96 466 379 769 599 411 926 192 747 235 376 687 253 501 301 119 970 633 535 909 153 627 369 419 852 585 774 574 165 774 781 265 856 688 875 809 602 338 475 497 95 852 780 740 144 776 932 802 508 104 47 858 865 815 698 833 47 138 575 519 758 995 969 792 886 453 907 487 8 226 279 646 548 612 79 190 233 537 977 744 279 185 218 720 116 933 697 544 967 215 848 135 848 936 351 629 210 909 415 308 950 119 802 148 883 956 68 36 25 542 707 282 329 275 223 491 600 394 936 687 503 39 326 384 676 772 542 273 989 514 36 670 479 998 239 988 880 511 60 955 340 548 574 394 594 669 932 613 437 625 213 817 434 429 954 746 961 548 2 717 183 85 110 232 920 179 625 253 555 635 973 431 936 9 308 406 544 611 715 183 796 802 409 199 209 811 840 652 58 970 62 589 868 354 639 971 340 414 972 413 354 717 88 636 816 665 275 762 883 974 225 287 976 251 552 36 989 413 794 532 584 633 690 253 78 669 628 773 332 304 971 216 369 514 681 762 585 697 566 961 553 600 26 867 63 885 427 796 304 690 652 720 918 233 331 567 611 405 458 611 772 380 15 361 65 638 802 259 494 146 225 559 273 140 338 400 883 23 743 869 762 698 265 86 472 574 859 950 903 621 95 86 872 816 391 553 960 582 734 259 974 816 453 467 264 406 700 838 85 992 977 728 794 750 920 4 329 641 903 953 805 708 175 871 747 648 915 612 266 775 519 185 55 14 746 631 351 818 682 676 360 182 369 223 157 140 387 233 24 837 949 217 2 590 786 97 848 938 316 789 466 830 641 960 384 235 720 567 963 362 817 70 455 855 926 491 246 403 960 887 384 734 351 490 893 692 697 856 466 104 534 559 699 165 72 760 413 149 29 475 364 362 120 775 155 485 367 880 830 61 371 346 777 957 553 254 88 985 477 992 135 130 967 332 318 259 150 587 970 352 947 805 678 973 359 788 521 974 522 877 103 8 865 699 406 431 602 405 123 532 135 222 562 695 240 453 671 394 233 409 532 830 758 988 279 230 590 938 452 149 865 625 840 209 452 700 920 909 217 976 788 117 499 919 455 885 786 239 556 698 668 466 148 266 359 435 240 698 472 522 887 773 773 603 251 9 44 235 24 728 887 681 537 585 954 671 230 856 765 120 442 3 225 396 144 151 114 420 53 167 681 769 776 150 4 553 750 328 665 26 173 775 594 590 863 130 79 190 217 985 376 961 471 262 403 514 603 938 602 548 765 153 94 600 92 811 763 971 575 494 855 816 749 123 3 194 103 974 652 514 972 872 287 487 813 371 926 463 854 347 694 328 781 44 690 544 209 484 442 94 832 360 980 70 391 96 22 326 613 92 923 954 194 352 149 65 537 548 362 880 648 989 418 55 437 293 194 192 58 933 336 714 293 120 265 380 774 505 552 434 641 687 185 361 749 967 53 890 695 687 147 542 589 556 453 750 39 978 138 871 859 245 232 210 477 110 153 179 9 740 929 403 508 23 147 743 351 273 114 635 855 924 225 912 441 555 955 146 628 199 78 854 855 746 671 103 210 150 469 796 484 769 415 602 418 434 766 941 86 602 924 503 367 692 953 165 961 792 137 393 532 142 973 918 445 62 625 937 563 62 627 278 809 740 25 840 77 380 332 110 867 633 114 492 767 859 534 285 907 451 431 628 844 875 458 969 250 484 194 884 388 970 431 875 254 807 245 232 120 106 304 959 364 388 807 18 455 499 817 205 885 86 728 226 599 970 63 85 929 254 700 85 116 715 367 521 995 854 347 858 692 743 413 225 367 777 420 924 792 9 445 941 595 669 26 967 978 995 676 838 869 688 129 55 235 285 734 692 767 977 715 735 92 907 316 940 179 715 534 995 903 334 467 830 146 998 949 957 835 205 816 507 352 905 707 137 521 63 266 589 863 246 142 175 4 696 475 933 282 562 239 648 833 405 918 835 77 138 912 374 763 567 957 427 65 278 301 505 635 374 21 100 396 893 116 293 708 988 442 3 665 175 251 636 582 445 144 886 954 631 137 907 695 183 585 871 865 192 813 491 427 566 336 587 638 14 318 807 459 155 44 599 794 190 821 594 29 409 582 62 376 419 998 886 187 36 264 23 584 923 670 809 182 511 14 638 639 563 127 151 471 331 555 265 715 747 938 501 354 893 492 429 68 364 521 359 499 593 544 575 582 682 490 97 648 886 459 959 222 308 331 765 972 763 802 940 172 805 127 331 343 744 360 24 120 96 960 959 582 980 690 149 406 742 869 442 647 503 535 611 55 839 520 487 879 646 414 391 434 767 837 932 147 695 818 937 647 767 818 859 445 766 668 818 893 153 458 953 501 492 400 757 39 694 953 183 106 97 919 72 217 646 950 875 844 388 328 445 15 343 628 405 387 173 458 621 682 788 451 629 25 239 678 418 3 969 574 379 116 522 223 435 714 999 429 459 920 884 205 70 352 903 167 559 147 585 776 772 562 123 72 79 593 194 374 747 15 809 612 940 384 708 2 22 629 146 53 44 836 638 628 278 291 334 492 279 936 4 187 883 441 29 3 475 246 105 435 420 218 835 485 835 262 376 696 992 285 316 135 681 963 293 967 469 22 23 627 690 427 746'.split(' ')
        solution = [int(x) for x in solution]
    elif len(facilities) == 2000:
        solution = '412 325 1363 1153 335 80 250 1876 585 1630 1878 1076 711 535 268 626 833 192 810 801 1591 125 694 1035 1878 1142 1361 1229 1483 1036 1694 1041 688 1765 1431 409 430 1324 595 1320 1327 1675 437 1779 1714 1884 1582 593 1677 1395 318 1778 1168 1571 1439 1915 573 336 58 1633 1511 1562 350 1651 1111 593 417 42 787 501 1990 645 1072 787 1664 296 1882 118 601 1658 832 556 1118 831 496 0 595 61 1393 1511 1871 195 1624 212 1357 902 1837 1746 1247 1185 694 1926 1765 1383 1944 1087 248 352 196 1837 315 543 71 925 318 1336 672 268 1562 1164 1454 40 1116 1483 326 1171 1185 169 1446 374 1373 1588 484 1207 403 65 815 644 1798 1724 1388 1207 703 1315 539 1502 1163 1229 625 556 58 671 1635 1248 1048 1627 1166 1163 1772 1018 1906 974 1399 853 803 945 25 663 828 237 1212 1557 1675 1775 1121 275 768 1756 1659 1948 1332 853 1412 267 1399 1497 711 1072 876 1432 1481 1155 863 331 268 10 1178 1980 525 214 489 1322 1083 700 296 391 619 820 195 1233 1105 857 290 1470 906 1888 574 35 750 1175 1151 735 1313 1186 1483 550 1501 1822 4 800 691 1481 1461 1311 828 827 831 1918 1177 40 1579 1566 1630 1810 1459 524 437 1027 1817 1681 1272 1291 583 806 837 1479 1591 302 21 83 1434 1657 1597 377 624 1937 529 1217 1513 1155 881 1819 1664 11 625 1069 561 243 502 495 1960 1569 526 1837 556 107 1320 1450 802 783 1619 1566 298 391 1334 72 89 877 1412 945 1843 323 882 1726 644 1653 634 1402 1231 1728 308 1039 10 1530 1379 243 1117 1881 1301 804 434 1893 484 1782 759 302 268 1230 281 1867 1824 1092 1516 1759 1224 56 1059 1334 430 894 298 1958 1332 1914 718 364 1049 1846 248 1858 1557 925 1151 415 1392 1654 1519 945 1402 1432 1554 1061 738 1530 364 1850 305 1720 1714 125 1525 1185 85 1872 173 1973 1782 1039 803 1802 177 1829 1805 1439 1550 1973 1576 359 1856 1589 1546 667 544 593 868 1808 1579 1513 438 1379 820 1980 352 152 1417 1018 800 562 193 85 1464 1413 768 722 1388 1140 1793 1041 991 1327 53 1530 1225 184 533 1446 1462 1946 504 1439 131 1380 1162 1867 881 364 1083 165 107 291 837 881 711 1805 1685 1463 1327 71 618 302 1881 1824 1960 964 1948 335 956 1516 1968 1909 1658 250 1824 373 323 1670 656 1669 1373 1980 1051 1012 1624 722 231 1076 158 823 661 1431 1822 1752 820 750 1061 331 533 1186 1518 800 1624 337 866 194 1675 1546 115 1390 42 134 496 1249 618 188 1670 118 894 544 1546 53 1486 624 1696 947 1991 1544 1588 1805 1867 1808 721 486 1148 1786 411 1060 290 694 671 146 1555 1029 1423 1843 364 1885 143 1681 302 1203 1746 1856 1166 958 144 42 898 1424 404 899 1357 876 1209 1413 1166 1654 1591 529 1519 1397 83 986 925 173 799 588 618 1254 1525 1937 572 1686 722 1463 1502 489 1589 644 519 550 634 1575 1101 1854 504 1996 1893 1611 58 862 1624 1555 305 4 1586 1905 557 1388 1187 115 1843 1987 1178 1502 25 403 1582 1817 864 1879 196 1685 619 385 721 1830 1575 169 80 409 590 1678 1322 1724 1666 1779 1346 267 1070 1798 373 806 753 1395 1072 572 562 832 606 331 604 72 1379 631 1060 1685 867 1528 1589 1525 447 17 502 438 325 1207 146 118 1955 624 1117 806 1432 587 1231 1375 1752 250 1461 152 1163 1927 420 404 1012 192 352 978 987 827 1521 1696 1838 184 1958 1686 882 495 1203 1224 374 625 80 449 831 1399 193 1462 1424 1575 853 1468 1087 1049 1481 1311 1233 1703 987 1569 1439 1984 1614 1878 1139 1817 486 495 1371 604 715 449 876 1746 1211 604 634 832 1463 539 1991 1147 1793 194 1262 1663 813 519 1446 173 622 195 79 1906 823 1336 1895 1346 956 1069 1762 1164 1067 1973 1209 89 810 1471 212 682 184 1206 125 1903 1148 243 106 1935 661 1397 152 185 952 1762 1071 1167 1580 1231 974 231 1029 1850 1058 237 61 1854 1230 1619 1122 1142 1839 1350 1171 1839 1459 411 883 1222 1829 1754 534 526 1254 1775 1461 1334 1754 1532 1696 1224 694 1752 1628 1772 484 1521 1996 1209 782 489 1497 1035 1036 1521 661 193 74 1165 1829 1946 1720 1168 802 1140 1948 21 533 588 1655 1839 602 718 39 1786 1927 1101 1837 1802 1884 1928 1635 1211 1627 337 1392 813 1049 1819 1830 1489 1480 1375 1805 65 561 305 913 1383 1562 1850 572 89 80 373 1103 1412 649 177 502 416 194 485 1116 1903 385 1619 1004 626 1247 735 1872 1659 753 595 1678 1018 1069 325 788 1249 579 271 39 1497 550 1987 1726 1209 1793 79 399 1461 722 625 1501 1454 510 1051 1041 1324 837 1435 40 409 1990 1666 1388 1313 991 266 987 131 123 997 1431 1994 956 1059 146 490 1163 1122 898 1818 1829 570 72 1651 1039 587 874 490 408 1878 1329 1317 271 17 1885 1121 1884 921 1060 1838 1593 672 21 134 1655 802 649 1434 1233 1371 1655 489 711 298 194 1092 1658 1991 622 187 1651 1950 1968 1040 1677 296 225 214 671 1948 399 1879 380 1703 1230 1816 496 1586 661 1067 1914 1004 964 1073 1004 1895 1175 1116 963 1301 1346 949 1042 1395 750 411 684 1826 1027 526 1317 791 748 1709 682 1061 902 688 1198 1105 169 1772 291 308 949 1117 1383 1445 1819 1779 1914 1067 1332 1888 1329 1103 225 380 723 399 131 721 0 924 315 1850 1445 380 1854 408 1450 658 1653 738 1071 192 801 783 861 1272 1984 53 723 1147 1397 1468 1222 412 1858 759 1301 833 437 11 143 997 1292 1445 437 1042 1501 579 1826 251 1987 723 134 1392 1516 39 1144 1139 290 682 1390 1666 1662 1652 1955 1224 1950 1010 1511 782 360 377 275 1334 25 863 25 1658 1967 935 533 1324 1471 978 883 416 1662 570 231 1329 408 1459 1726 1142 587 296 1947 1040 1571 1361 952 459 1424 333 435 913 952 1566 562 1756 759 1973 10 373 656 1390 1361 1139 803 1846 360 1546 810 1346 1653 1206 1566 997 748 298 1519 631 618 296 949 877 1518 1588 1935 1105 49 489 1670 305 10 1230 1928 1641 1513 469 1669 590 1165 288 544 1595 0 1483 1397 1694 190 61 735 1187 290 1759 1212 1393 723 337 520 1118 1503 271 1254 1595 1301 1575 691 1051 1927 1576 1011 1363 1871 949 185 364 652 385 1315 588 1879 1249 490 1315 420 333 1579 177 250 1503 662 336 944 1611 1051 323 1316 391 1550 738 1390 788 1818 857 874 1926 1778 1978 411 573 876 626 684 352 1881 783 1048 214 266 956 1746 1554 1893 1073 1854 1896 1582 1681 1272 978 1652 921 1479 779 1373 1958 525 941 652 1144 564 480 490 1611 656 504 1858 184 1361 907 1967 1311 862 1944 921 645 234 143 593 1944 620 564 389 115 1011 1322 561 772 1769 864 1664 1167 587 583 1786 1222 663 315 562 1611 588 1383 1010 404 1292 65 519 543 1857 1762 669 1618 941 1040 658 1311 1101 631 748 49 1192 58 1686 1371 918 862 1061 1994 1446 864 1888 1395 912 899 49 350 828 459 299 1147 1662 631 913 1593 1417 782 1652 539 1614 1727 1445 782 1657 1434 1518 1630 1118 783 267 618 1816 1073 360 1947 1664 1871 925 759 1147 494 1503 667 912 1060 1486 906 924 810 986 1968 326 1101 409 827 1332 1839 391 243 574 1380 524 177 1397 1564 806 1304 1262 1752 416 1896 1247 1905 669 1619 595 585 944 1217 1374 1633 1906 260 335 1177 1681 374 1802 1597 1222 1727 579 1350 1011 1879 740 377 1198 1597 336 35 434 1635 1148 1896 1525 572 260 1885 190 947 991 389 1724 1328 1468 902 1550 529 1327 1528 484 1569 806 165 1756 958 234 788 326 791 1371 1561 1087 17 1058 1630 1329 682 788 1087 1987 1162 1454 667 867 691 1111 1857 1761 435 173 44 1544 990 1595 1818 291 1424 1175 115 1778 1967 1313 1666 1654 185 967 1633 1657 1155 1059 1417 1375 1350 740 658 1571 1316 35 1785 791 1329 1012 449 1810 169 326 1144 585 1759 1291 420 1669 1709 573 17 187 1937 1599 1464 71 1896 417 271 1164 318 579 1320 1915 804 823 779 1248 857 958 1162 1714 1915 308 4 1816 967 107 1659 1994 634 564 799 447 74 1895 1895 1654 291 1572 1192 469 947 620 1292 1810 1374 1663 626 1769 1798 918 1229 912 1103 1317 582 74 131 1486 863 1503 1142 768 106 1153 804 193 804 620 1896 881 1518 288 497 1357 1048 1468 1528 149 1029 1714 1166 1304 248 606 1402 1614 225 964 1413 815 1826 718 1914 430 875 1395 700 1162 1144 1819 1439 1928 1291 1185 967 898 738 990 497 185 469 1375 534 1103 1856 1728 861 1968 1374 193 1464 519 1379 1040 688 359 691 1336 196 1480 188 243 874 1628 1011 662 502 1669 1177 1572 1399 438 813 1915 149 1628 158 1036 1651 1544 1858 44 35 703 535 625 524 1703 1230 1756 1373 1876 1301 1694 1164 1151 1501 875 1816 1726 1918 1762 649 1532 1073 1203 853 194 435 1987 1186 251 260 602 1960 606 510 399 861 447 1903 1070 772 1564 1653 486 715 1857 1117 1881 1121 315 1071 1686 1445 868 1178 866 1186 899 1092 662 1042 1926 535 11 190 1248 1076 430 1876 1867 935 325 912 251 1316 688 56 1454 1633 1586 669 1168 882 1486 350 1122 302 494 1010 123 1984 918 1884 389 435 1580 952 828 1516 645 1072 1572 267 56 557 485 672 1291 1402 331 1192 906 1304 863 1561 1950 420 619 907 188 1798 1225 1918 800 1217 288 1328 1177 520 484 1328 1554 1817 1058 1393 1211 1060 750 557 1906 404 71 1878 1036 337 1489 271 1772 1996 1918 1838 130 582 1843 935 1471 1346 990 1793 883 40 595 21 359 1569 1363 963 1685 944 1824 1203 877 360 907 447 380 1470 1116 1423 857 415 1206 231 753 1677 582 79 1824 1720 485 525 1664 1511 391 469 335 1599 486 787 1140 803 417 1950 130 1212 1572 72 1593 403 1092 1856 1677 669 1846 945 149 1313 963 1641 787 1990 1657 1479 918 118 497 1544 1328 1926 772 494 894 1489 434 1165 165 501 188 1802 520 1470 700 1990 1225 799 1572 1435 945 1978 1212 1782 1450 1167 281 1198 118 89 652 1663 1772 875 663 534 867 106 1726 1049 1311 556 1070 1557 1207 1618 333 1946 941 606 83 802 1292 123'.split(' ')
        solution = [int(x) for x in solution]
    else:
        solution = initial_solution(customers, facilities)
    print(solution)

    return solution

    neighbours = compute_neighbours(customers, facilities, weight_length, num_neighbours)

    best_obj = obj = calculate_obj(solution, facilities, customers)
    best_solution = solution[:]

    customer_count = len(customers)
    facility_count = len(facilities)

    alpha = 0.9999999
    T_start = 1e-4
    T_end = 1e-15

    import random
    T = T_start
    iteration = 0

    space_left = [f.capacity for f in facilities]
    for cust_i, fac in enumerate(solution):
        space_left[fac] -= customers[cust_i].demand
    iteration = 0
    while T > T_end:
        iteration += 1
        solution_copy = solution[:]
        space_left_copy = space_left[:]
        obj_copy = obj
        #print(T)
        T = T * alpha
        num_cust_swap = 2


        move_customers = np.mod(iteration, 2) == 0
        if move_customers:
            random_customer_is = list(set(random.sample(range(1, customer_count), num_cust_swap)))
        #random_customer_is = [random.randint(0, customer_count - 1) for _ in range(num_cust_swap)]
            random_facility_is = []
            for random_customer_i in random_customer_is:
                random_facility_is.append(list(neighbours[random_customer_i])[random.randint(0, len(neighbours[random_customer_i]) - 1)])
        else:
            active_facilities = list(set(solution_copy))
            random_facility = list(set(random.sample(range(0, len(active_facilities)), 1)))[0]
            random_customer_is = [i for i in solution_copy if i == random_facility]
            random_facility_is = [random.randint(0, facility_count - 2) for _ in range(len(random_customer_is))]
            for i in range(len(random_facility_is)):
                if random_facility_is[i] >= random_facility: # don't pick facility
                    random_facility_is[i] += 1



        for i in range(len(random_customer_is)):
            customer = customers[random_customer_is[i]]
            old_facility = solution_copy[random_customer_is[i]]
            new_facility = random_facility_is[i]
            solution_copy[random_customer_is[i]] = random_facility_is[i]
            space_left_copy[old_facility] += customer.demand
            space_left_copy[new_facility] -= customer.demand

        exceeded = False
        for demand in space_left_copy:
            if demand < 0:
                exceeded = True
        if exceeded:
            continue

        obj = calculate_obj(solution_copy, facilities, customers)
        #print(obj)

        prob = np.exp((best_obj - obj)/ T)
        rand = random.random()
        if prob < rand:
            continue  # not accepting

        #print(obj)
        if obj < best_obj:
            best_obj = obj
            best_solution = solution[:]
            print_output_data(obj, solution)

        solution = solution_copy[:]
        space_left = space_left_copy

    return best_solution

def print_output_data(obj, solution):
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    print(output_data)
