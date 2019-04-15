/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <list>
#include <type_traits>
#include <utility>
#include <array>
#include <vector>
#include <thread>
#include <algorithm>
#include <prototype/generic_interfaces.hpp>
#include <prototype/regular_grid_descriptors.hpp>
#include <prototype/triplet.hpp>

namespace gt = gridtools;


namespace test_data_descriptor {


    template <int N, typename T>
    void show_data(T x, std::ostream& s = std::cout) {
        s << "Data:\n";
        for (int i = 0; i<N; ++i) {
            for (int j = 0; j<N; ++j) {
                s << x[i][j] << ", "; // TO_DO: change access operator
            }
            s << "\n";
        }
        s << "\n";
        s.flush();
    }


    template<int rank, typename DA> // type of data array is now templated (DA), as well as rank
    class data_descriptor_t {

      private:

        struct range {

            int m_begin;
            int m_end;

            range(int b, int e) : m_begin(b), m_end(e) {}

            int begin() const {
                return m_begin;
            }

            int end() const {
                return m_end;
            }

        };

        std::array<int, rank> m_sizes;
        DA m_values;

      public:

        template <typename ...Sizes>
        data_descriptor_t(const DA& values, Sizes... s) : m_sizes{s...}, m_values{values} {}

        template <int I>
        int begin() const {
            return 0;
        }

        template <int I>
        int end() const {
            return m_sizes[I];
        }

        template <int I>
        range range_of() const {
            return range(begin<I>(), end<I>());
        }

        template <typename DT, typename ...Coords> // data type is templated
        DT get_value(Coords... cs) const {
            return m_values(cs...);
        }

    };


}


int main(int argc, char** argv) {

    typedef double T1;
    typedef gridtools::layout_map<0, 1, 2> layoutmap;

    int coords[3] = {0, 1, 2};

    const int DIM1 = 3, DIM2 = 3, DIM3 = 3;
    const int H1m = 1, H1p = 1, H2m = 1, H2p = 1, H3m = 1, H3p = 1;

    triple_t<USE_DOUBLE, T1> * _a = new triple_t<USE_DOUBLE, T1>[(DIM1 + H1m + H1p) * (DIM2 + H2m + H2p) * (DIM3 + H3m + H3p)];
    array<triple_t<USE_DOUBLE, T1>, layoutmap> a(_a, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p));

    /* array initialization */
    for (int ii = 0; ii < DIM1 + H1m + H1p; ++ii) {
        for (int jj = 0; jj < DIM2 + H2m + H2p; ++jj) {
            for (int kk = 0; kk < DIM3 + H3m + H3p; ++kk) {
                a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>();
            }
        }
    }

    /* array initialization */
    for (int ii = H1m; ii < DIM1 + H1m; ++ii) {
        for (int jj = H2m; jj < DIM2 + H2m; ++jj) {
            for (int kk = H3m; kk < DIM3 + H3m; ++kk) {
                a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>(
                    ii - H1m + (DIM1)*coords[0], jj - H2m + (DIM2)*coords[1], kk - H3m + (DIM3)*coords[2]);
            }
        }
    }

    std::cout << "A \n";
    printbuff(std::cout, a, DIM1 + H1m + H1p, DIM2 + H2m + H2p, DIM3 + H3m + H3p);

    test_data_descriptor::data_descriptor_t<3, array<triple_t<USE_DOUBLE, T1>, layoutmap>> data{a, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p)};

    auto data_1_1_1 = data.get_value<triple_t<USE_DOUBLE, T1>>(1, 1, 1);
    std::cout << "A(1, 1, 1) = (" << data_1_1_1.x() << ", " << data_1_1_1.y() << ", " << data_1_1_1.z() << ")" << std::endl;

}
