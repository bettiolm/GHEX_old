/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <sstream>
#include <fstream>

#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <string>
#include <utility>

#include <type_traits>
#include <list>
#include <array>
#include <vector>
#include <algorithm>

#include <gridtools/common/boollist.hpp>
#include <gridtools/meta/utility.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>

#include <prototype/generic_interfaces.hpp>
#include <prototype/range_loops.hpp>
#include <prototype/regular_grid_descriptors.hpp>
#include <prototype/triplet.hpp>

namespace gt = gridtools;

using id_type = std::array<int,3>;


std::ostream& operator<<(std::ostream& os, const id_type& x) {
    os << "(" << x[0] << ", " << x[1] << ", " << x[2] << ")";
    return os;
}


namespace std {
    template<> struct hash<id_type> {
        std::size_t operator()(id_type const& t) const {
            return std::hash<int>{}(t[0]); // To DO: find better hash function!
        }
    };
}


struct dir_type : public gt::direction<3> {

    using gt::direction<3>::direction;

    static int direction2int(dir_type d) {
        return d.m_data[0]*3*3+d.m_data[1]*3+d.m_data[2] + 13;
    }

    static dir_type invert_direction(dir_type d) {
        return dir_type{std::array<int, 3>{-d.m_data[0], -d.m_data[1], -d.m_data[2]}};
    }

};


template<int r, typename DA, typename DT>
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

    std::array<int, r> m_sizes;
    DA m_data;

  public:

    static const int rank = r;

    template <typename ...Sizes>
    data_descriptor_t(const DA& data, Sizes... s) : m_sizes{s...}, m_data{data} {}

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

    DT get_data(std::array<int, r> indices) const {
        return get_data(indices, gridtools::meta::make_integer_sequence<int, r>{});
    }

    void set_data(DT value, std::array<int, r> indices) {
        set_data(value, indices, gridtools::meta::make_integer_sequence<int, r>{});
    }

    void show_data(std::ostream& s) {
        auto ranges_of_data = make_range_of_data(gridtools::meta::make_integer_sequence<int, r>{});
        gridtools::range_loop(ranges_of_data, [&s, this](auto const& indices) {
            show_data(s, indices);
            s << " "; // TO DO: still misisng a way to format the data nicely
        });
    }

  private:

    template <int... Dims>
    DT get_data(std::array<int, r> indices, gridtools::meta::integer_sequence<int, Dims...>) const {
        return m_data(indices[Dims]...);
    }

    template <int... Dims>
    void set_data(DT value, std::array<int, r> indices, gridtools::meta::integer_sequence<int, Dims...>) {
        m_data(indices[Dims]...) = value;
    }

    template <int... Dims>
    void show_data(std::ostream& s, std::array<int, r> indices, gridtools::meta::integer_sequence<int, Dims...>) const {
        s << m_data(indices[Dims]...);
    }

    void show_data(std::ostream& s, std::array<int, r> indices) const {
        show_data(s, indices, gridtools::meta::make_integer_sequence<int, r>{});
    }

    template <int... Dims>
    auto make_range_of_data(gridtools::meta::integer_sequence<int, Dims...>) {
        return std::make_tuple(range_of<Dims>()...);
    }

};


namespace halo_exchange_3D_generic_full {
    int pid;
    int nprocs;
    MPI_Comm CartComm;
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};

    struct timeval start_tv;
    struct timeval stop1_tv;
    struct timeval stop2_tv;
    struct timeval stop3_tv;
    double lapse_time1;
    double lapse_time2;
    double lapse_time3;
    double lapse_time4;

#define B_ADD 1
#define C_ADD 2

    typedef int T1;
    typedef double T2;
    typedef long long int T3;

    template <typename ST, int I1, int I2, int I3, bool per0, bool per1, bool per2>
    bool run(ST &file,
        int DIM1,
        int DIM2,
        int DIM3,
        int H1m1,
        int H1p1,
        int H2m1,
        int H2p1,
        int H3m1,
        int H3p1,
        int H1m2,
        int H1p2,
        int H2m2,
        int H2p2,
        int H3m2,
        int H3p2,
        int H1m3,
        int H1p3,
        int H2m3,
        int H2p3,
        int H3m3,
        int H3p3,
        triple_t<USE_DOUBLE, T1> *_a,
        triple_t<USE_DOUBLE, T2> *_b,
        triple_t<USE_DOUBLE, T3> *_c) {

        typedef gridtools::layout_map<I1, I2, I3> layoutmap;

        typedef data_descriptor_t<3, array<triple_t<USE_DOUBLE, T1>, layoutmap>, triple_t<USE_DOUBLE, T1>> data_dsc_type_1;
        typedef data_descriptor_t<3, array<triple_t<USE_DOUBLE, T2>, layoutmap>, triple_t<USE_DOUBLE, T2>> data_dsc_type_2;
        typedef data_descriptor_t<3, array<triple_t<USE_DOUBLE, T3>, layoutmap>, triple_t<USE_DOUBLE, T3>> data_dsc_type_3;

        id_type id{coords[0], coords[1], coords[2]};
        std::list<id_type> local_ids{id};

        auto neighbor_generator = [](id_type _id) -> std::vector<std::pair<id_type, dir_type>> {

            int i = _id[0];
            int j = _id[1];
            int k = _id[2];
            std::vector<std::pair<int, int>> iis{std::make_pair(0,i)}, jjs{std::make_pair(0,j)}, kks{std::make_pair(0,k)};
            std::vector<std::pair<id_type, dir_type>> neighbors;

            if (per0) {
                iis.push_back(std::make_pair(-1, mod(i-1, dims[0])));
                iis.push_back(std::make_pair(1, mod(i+1, dims[0])));
            } else {
                if (i > 0) iis.push_back(std::make_pair(-1, i-1));
                if (i < (dims[0]-1)) iis.push_back(std::make_pair(1, i+1));
            }
            if (per1) {
                jjs.push_back(std::make_pair(-1, mod(j-1, dims[1])));
                jjs.push_back(std::make_pair(1, mod(j+1, dims[1])));
            } else {
                if (j > 0) jjs.push_back(std::make_pair(-1, j-1));
                if (j < (dims[1]-1)) jjs.push_back(std::make_pair(1, j+1));
            }
            if (per2) {
                kks.push_back(std::make_pair(-1, mod(k-1, dims[2])));
                kks.push_back(std::make_pair(1, mod(k+1, dims[2])));
            } else {
                if (k > 0) kks.push_back(std::make_pair(-1, k-1));
                if (k < (dims[2]-1)) kks.push_back(std::make_pair(1, k+1));
            }

            for (auto ii : iis) {
                for (auto jj : jjs) {
                    for (auto kk : kks) {
                        if (!(ii.first == 0 && jj.first == 0 && kk.first == 0)) {
                            neighbors.push_back(std::make_pair(id_type{ii.second, jj.second, kk.second}, std::array<int, 3>{ii.first, jj.first, kk.first}));
                        }
                    }
                }
            };

            return neighbors;

        };

        file << "Local ids\n";
        std::for_each(local_ids.begin(), local_ids.end(), [&file, neighbor_generator] (id_type const& x) {
            auto list = neighbor_generator(x);
            file << "neighbors of ID = " << x << ":\n";
            std::for_each(list.begin(), list.end(), [&file](std::pair<id_type, dir_type> const& y) {
                file << y.first << ", ";
            });
            file << "\n";
        });
        file << "\n";

        generic_pg<id_type, dir_type> pg(local_ids, neighbor_generator, file);

        pg.show_topology(file);
        file.flush();

        array<triple_t<USE_DOUBLE, T1>, layoutmap> a(
            _a, (DIM1 + H1m1 + H1p1), (DIM2 + H2m1 + H2p1), (DIM3 + H3m1 + H3p1));
        array<triple_t<USE_DOUBLE, T2>, layoutmap> b(
            _b, (DIM1 + H1m2 + H1p2), (DIM2 + H2m2 + H2p2), (DIM3 + H3m2 + H3p2));
        array<triple_t<USE_DOUBLE, T3>, layoutmap> c(
            _c, (DIM1 + H1m3 + H1p3), (DIM2 + H2m3 + H2p3), (DIM3 + H3m3 + H3p3));

        file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";

        /* Just an initialization */
        for (int ii = 0; ii < DIM1 + H1m1 + H1p1; ++ii)
            for (int jj = 0; jj < DIM2 + H2m1 + H2p1; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m1 + H3p1; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>();
                }
            }

        for (int ii = 0; ii < DIM1 + H1m2 + H1p2; ++ii)
            for (int jj = 0; jj < DIM2 + H2m2 + H2p2; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m2 + H3p2; ++kk) {
                    b(ii, jj, kk) = triple_t<USE_DOUBLE, T2>();
                }
            }

        for (int ii = 0; ii < DIM1 + H1m3 + H1p3; ++ii)
            for (int jj = 0; jj < DIM2 + H2m3 + H2p3; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m3 + H3p3; ++kk) {
                    c(ii, jj, kk) = triple_t<USE_DOUBLE, T3>();
                }
            }

        for (int ii = H1m1; ii < DIM1 + H1m1; ++ii)
            for (int jj = H2m1; jj < DIM2 + H2m1; ++jj)
                for (int kk = H3m1; kk < DIM3 + H3m1; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>(
                        ii - H1m1 + (DIM1)*coords[0], jj - H2m1 + (DIM2)*coords[1], kk - H3m1 + (DIM3)*coords[2]);
                }

        for (int ii = H1m2; ii < DIM1 + H1m2; ++ii)
            for (int jj = H2m2; jj < DIM2 + H2m2; ++jj)
                for (int kk = H3m2; kk < DIM3 + H3m2; ++kk) {
                    b(ii, jj, kk) = triple_t<USE_DOUBLE, T2>(ii - H1m2 + (DIM1)*coords[0] + B_ADD,
                        jj - H2m2 + (DIM2)*coords[1] + B_ADD,
                        kk - H3m2 + (DIM3)*coords[2] + B_ADD);
                }

        for (int ii = H1m3; ii < DIM1 + H1m3; ++ii)
            for (int jj = H2m3; jj < DIM2 + H2m3; ++jj)
                for (int kk = H3m3; kk < DIM3 + H3m3; ++kk) {
                    c(ii, jj, kk) = triple_t<USE_DOUBLE, T3>(ii - H1m3 + (DIM1)*coords[0] + C_ADD,
                        jj - H2m3 + (DIM2)*coords[1] + C_ADD,
                        kk - H3m3 + (DIM3)*coords[2] + C_ADD);
                }

        file << "A \n";
        printbuff(file, a, DIM1 + H1m1 + H1p1, DIM2 + H2m1 + H2p1, DIM3 + H3m1 + H3p1);
        file << "B \n";
        printbuff(file, b, DIM1 + H1m2 + H1p2, DIM2 + H2m2 + H2p2, DIM3 + H3m2 + H3p2);
        file << "C \n";
        printbuff(file, c, DIM1 + H1m3 + H1p3, DIM2 + H2m3 + H2p3, DIM3 + H3m3 + H3p3);
        file.flush();

        data_dsc_type_1 data_dsc_a{a, DIM1 + H1m1 + H1p1, DIM2 + H2m1 + H2p1, DIM3 + H3m1 + H3p1};
        data_dsc_type_2 data_dsc_b{b, DIM1 + H1m2 + H1p2, DIM2 + H2m2 + H2p2, DIM3 + H3m2 + H3p2};
        data_dsc_type_3 data_dsc_c{c, DIM1 + H1m3 + H1p3, DIM2 + H2m3 + H2p3, DIM3 + H3m3 + H3p3};

        std::array<gt::halo_sizes, 3> halos_a = {gt::halo_sizes{H1m1, H1p1}, gt::halo_sizes{H2m1, H2p1}, gt::halo_sizes{H3m1, H3p1}};
        std::array<gt::halo_sizes, 3> halos_b = {gt::halo_sizes{H1m2, H1p2}, gt::halo_sizes{H2m2, H2p2}, gt::halo_sizes{H3m2, H3p2}};
        std::array<gt::halo_sizes, 3> halos_c = {gt::halo_sizes{H1m3, H1p3}, gt::halo_sizes{H2m3, H2p3}, gt::halo_sizes{H3m3, H3p3}};

        gt::regular_grid_descriptor<3> grid_a(halos_a);
        gt::regular_grid_descriptor<3> grid_b(halos_b);
        gt::regular_grid_descriptor<3> grid_c(halos_c);

        auto iteration_spaces_send_a = [&data_dsc_a, &grid_a](id_type local, id_type remote, dir_type direction) {
            return grid_a.inner_iteration_space<gt::partitioned<0, 1, 2>>(data_dsc_a, direction);
        };

        auto iteration_spaces_recv_a = [&data_dsc_a, &grid_a](id_type local, id_type remote, dir_type direction) {
            return grid_a.outer_iteration_space<gt::partitioned<0, 1, 2>>(data_dsc_a, direction);
        };

        auto iteration_spaces_send_b = [&data_dsc_b, &grid_b](id_type local, id_type remote, dir_type direction) {
            return grid_b.inner_iteration_space<gt::partitioned<0, 1, 2>>(data_dsc_b, direction);
        };

        auto iteration_spaces_recv_b = [&data_dsc_b, &grid_b](id_type local, id_type remote, dir_type direction) {
            return grid_b.outer_iteration_space<gt::partitioned<0, 1, 2>>(data_dsc_b, direction);
        };

        auto iteration_spaces_send_c = [&data_dsc_c, &grid_c](id_type local, id_type remote, dir_type direction) {
            return grid_c.inner_iteration_space<gt::partitioned<0, 1, 2>>(data_dsc_c, direction);
        };

        auto iteration_spaces_recv_c = [&data_dsc_c, &grid_c](id_type local, id_type remote, dir_type direction) {
            return grid_c.outer_iteration_space<gt::partitioned<0, 1, 2>>(data_dsc_c, direction);
        };

        typedef generic_co<generic_pg<id_type, dir_type>, decltype(iteration_spaces_send_a), decltype(iteration_spaces_recv_a)> co_type_a;
        typedef generic_co<generic_pg<id_type, dir_type>, decltype(iteration_spaces_send_b), decltype(iteration_spaces_recv_b)> co_type_b;
        typedef generic_co<generic_pg<id_type, dir_type>, decltype(iteration_spaces_send_c), decltype(iteration_spaces_recv_c)> co_type_c;

        std::vector<co_type_a> co_a;
        std::vector<co_type_b> co_b;
        std::vector<co_type_c> co_c;

        for (auto id : local_ids) {
            co_a.push_back(co_type_a{id, pg, iteration_spaces_send_a, iteration_spaces_recv_a});
            co_b.push_back(co_type_b{id, pg, iteration_spaces_send_b, iteration_spaces_recv_b});
            co_c.push_back(co_type_c{id, pg, iteration_spaces_send_c, iteration_spaces_recv_c});
        }

        auto itc_a = co_a.begin();
        auto itc_b = co_b.begin();
        auto itc_c = co_c.begin();

        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&start_tv, nullptr);

        /* TO DO: debugging string to be removed */
        for (auto it = local_ids.begin(); it != local_ids.end(); ++it, ++itc_a, ++itc_b, ++itc_c) {

            std::stringstream ss;
            ss << pid;
            std::string filename = "tout" + ss.str() + ".txt";
            std::ofstream tfile(filename.c_str());
            tfile << "\nFILE for " << *it << "\n";

            auto hdl_a = (*itc_a).template exchange<data_dsc_type_1, triple_t<USE_DOUBLE, T1>>(data_dsc_a, tfile);
            auto hdl_b = (*itc_b).template exchange<data_dsc_type_2, triple_t<USE_DOUBLE, T2>>(data_dsc_b, tfile);
            auto hdl_c = (*itc_c).template exchange<data_dsc_type_3, triple_t<USE_DOUBLE, T3>>(data_dsc_c, tfile);

            hdl_a.wait();
            hdl_b.wait();
            hdl_c.wait();

            tfile.flush();
            tfile.close();
        }

        gettimeofday(&stop1_tv, nullptr);

        lapse_time1 =
            ((static_cast<double>(stop1_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop1_tv.tv_usec)) -
                (static_cast<double>(start_tv.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_tv.tv_usec))) *
            1000.0;

        MPI_Barrier(MPI_COMM_WORLD);

        file << "TIME TOT : " << lapse_time1 << std::endl;

        file << "\n********************************************************************************\n";

        file << "A \n";
        printbuff(file, a, DIM1 + H1m1 + H1p1, DIM2 + H2m1 + H2p1, DIM3 + H3m1 + H3p1);
        file << "B \n";
        printbuff(file, b, DIM1 + H1m2 + H1p2, DIM2 + H2m2 + H2p2, DIM3 + H3m2 + H3p2);
        file << "C \n";
        printbuff(file, c, DIM1 + H1m3 + H1p3, DIM2 + H2m3 + H2p3, DIM3 + H3m3 + H3p3);
        file.flush();

        int passed = true;

        /* Checking the data arrived correctly in the whole region
         */
        for (int ii = 0; ii < DIM1 + H1m1 + H1p1; ++ii)
            for (int jj = 0; jj < DIM2 + H2m1 + H2p1; ++jj)
                for (int kk = 0; kk < DIM3 + H3m1 + H3p1; ++kk) {

                    triple_t<USE_DOUBLE, T1> ta;
                    int tax, tay, taz;

                    tax = modulus(ii - H1m1 + (DIM1)*coords[0], DIM1 * dims[0]);

                    tay = modulus(jj - H2m1 + (DIM2)*coords[1], DIM2 * dims[1]);

                    taz = modulus(kk - H3m1 + (DIM3)*coords[2], DIM3 * dims[2]);

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m1)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m1))) {
                            tax = triple_t<USE_DOUBLE, T1>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m1)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m1))) {
                            tay = triple_t<USE_DOUBLE, T1>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m1)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m1))) {
                            taz = triple_t<USE_DOUBLE, T1>().z();
                        }
                    }

                    ta = triple_t<USE_DOUBLE, T1>(tax, tay, taz).floor();

                    if (a(ii, jj, kk) != ta) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "a " << a(ii, jj, kk) << " != " << ta << "\n";
                    }
                }

        for (int ii = 0; ii < DIM1 + H1m2 + H1p2; ++ii)
            for (int jj = 0; jj < DIM2 + H2m2 + H2p2; ++jj)
                for (int kk = 0; kk < DIM3 + H3m2 + H3p2; ++kk) {

                    triple_t<USE_DOUBLE, T2> tb;
                    int tbx, tby, tbz;

                    tbx = modulus(ii - H1m2 + (DIM1)*coords[0], DIM1 * dims[0]) + B_ADD;

                    tby = modulus(jj - H2m2 + (DIM2)*coords[1], DIM2 * dims[1]) + B_ADD;

                    tbz = modulus(kk - H3m2 + (DIM3)*coords[2], DIM3 * dims[2]) + B_ADD;

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m2)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m2))) {
                            tbx = triple_t<USE_DOUBLE, T2>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m2)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m2))) {
                            tby = triple_t<USE_DOUBLE, T2>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m2)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m2))) {
                            tbz = triple_t<USE_DOUBLE, T2>().z();
                        }
                    }

                    tb = triple_t<USE_DOUBLE, T2>(tbx, tby, tbz).floor();

                    if (b(ii, jj, kk) != tb) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "b " << b(ii, jj, kk) << " != " << tb << "\n";
                    }
                }

        for (int ii = 0; ii < DIM1 + H1m3 + H1p3; ++ii)
            for (int jj = 0; jj < DIM2 + H2m3 + H2p3; ++jj)
                for (int kk = 0; kk < DIM3 + H3m3 + H3p3; ++kk) {

                    triple_t<USE_DOUBLE, T3> tc;
                    int tcx, tcy, tcz;

                    tcx = modulus(ii - H1m3 + (DIM1)*coords[0], DIM1 * dims[0]) + C_ADD;

                    tcy = modulus(jj - H2m3 + (DIM2)*coords[1], DIM2 * dims[1]) + C_ADD;

                    tcz = modulus(kk - H3m3 + (DIM3)*coords[2], DIM3 * dims[2]) + C_ADD;

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m3)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m3))) {
                            tcx = triple_t<USE_DOUBLE, T3>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m3)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m3))) {
                            tcy = triple_t<USE_DOUBLE, T3>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m3)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m3))) {
                            tcz = triple_t<USE_DOUBLE, T3>().z();
                        }
                    }

                    tc = triple_t<USE_DOUBLE, T3>(tcx, tcy, tcz).floor();

                    if (c(ii, jj, kk) != tc) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "c " << c(ii, jj, kk) << " != " << tc << "\n";
                    }
                }

        if (passed)
            file << "RESULT: PASSED!\n";
        else
            file << "RESULT: FAILED!\n";

        return passed;
    }

    bool test(int DIM1,
        int DIM2,
        int DIM3,
        int H1m1,
        int H1p1,
        int H2m1,
        int H2p1,
        int H3m1,
        int H3p1,
        int H1m2,
        int H1p2,
        int H2m2,
        int H2p2,
        int H3m2,
        int H3p2,
        int H1m3,
        int H1p3,
        int H2m3,
        int H2p3,
        int H3m3,
        int H3p3) {

        /* Here we compute the computing grid as in many applications
         */
        MPI_Comm_rank(MPI_COMM_WORLD, &pid);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

        std::cout << pid << " " << nprocs << "\n";

        std::stringstream ss;
        ss << pid;

        std::string filename = "out" + ss.str() + ".txt";

        std::cout << filename << std::endl;
        std::ofstream file(filename.c_str());

        file << pid << "  " << nprocs << "\n";

        MPI_Dims_create(nprocs, 3, dims);
        int period[3] = {1, 1, 1};

        file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

        MPI_Cart_get(CartComm, 3, dims, period, coords);

        /* Each process will hold a tile of size
           (DIM1+2*H)x(DIM2+2*H)x(DIM3+2*H). The DIM1xDIM2xDIM3 area inside
           the H width border is the inner region of an hypothetical stencil
           computation whise halo width is H.
        */

        file << "Field A "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m1 << " - " << H1p1 << ", "
             << "Halo along j " << H2m1 << " - " << H2p1 << ", "
             << "Halo along k " << H3m1 << " - " << H3p1 << std::endl;

        file << "Field B "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m2 << " - " << H1p2 << ", "
             << "Halo along j " << H2m2 << " - " << H2p2 << ", "
             << "Halo along k " << H3m2 << " - " << H3p2 << std::endl;

        file << "Field C "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m3 << " - " << H1p3 << ", "
             << "Halo along j " << H2m3 << " - " << H2p3 << ", "
             << "Halo along k " << H3m3 << " - " << H3p3 << std::endl;
        file.flush();

        /* This example will exchange 3 data arrays at the same time with
           different values.
        */
        triple_t<USE_DOUBLE, T1> *_a =
            new triple_t<USE_DOUBLE, T1>[(DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1)];
        triple_t<USE_DOUBLE, T2> *_b =
            new triple_t<USE_DOUBLE, T2>[(DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2)];
        triple_t<USE_DOUBLE, T3> *_c =
            new triple_t<USE_DOUBLE, T3>[(DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3)];

        file << "Permutation 0,1,2\n";

        file << "run<std::ostream, 0,1,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";

        bool passed = true;

        passed = passed and run<std::ostream, 0, 1, 2, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,1,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,1,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,1,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,1,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,1,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,1,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 0,2,1\n";

        file << "run<std::ostream, 0,2,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,2,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,2,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 0,2,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 0,2,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,0,2\n";

        file << "run<std::ostream, 1,0,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,0,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,0,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,0,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,0,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,2,0\n";

        file << "run<std::ostream, 1,2,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,2,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,2,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 1,2,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 1,2,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H31, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,0,1\n";

        file << "run<std::ostream, 2,0,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,0,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,0,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,0,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,0,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,1,0\n";

        file << "run<std::ostream, 2,1,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,1,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,1,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file
            << "run<std::ostream, 2,1,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, true>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c);

        file << "run<std::ostream, 2,1,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, false>(file,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,

                                _a,
                                _b,
                                _c);
        file << "---------------------------------------------------\n";

        delete[] _a;
        delete[] _b;
        delete[] _c;

        return passed;
    }
} // namespace halo_exchange_3D_generic_full


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    if (argc != 22) {
        std::cout << "Usage: test_halo_exchange_3D dimx dimy dimz h1m1 hip1 h2m1 h2m1 h3m1 h3p1 h1m2 hip2 h2m2 h2m2 "
                     "h3m2 h3p2 h1m3 hip3 h2m3 h2m3 h3m3 h3p3\n where args are integer sizes of the data fields and "
                     "halo width"
                  << std::endl;
        return 1;
    }
    int DIM1 = atoi(argv[1]);
    int DIM2 = atoi(argv[2]);
    int DIM3 = atoi(argv[3]);
    int H1m1 = atoi(argv[4]);
    int H1p1 = atoi(argv[5]);
    int H2m1 = atoi(argv[6]);
    int H2p1 = atoi(argv[7]);
    int H3m1 = atoi(argv[8]);
    int H3p1 = atoi(argv[9]);
    int H1m2 = atoi(argv[10]);
    int H1p2 = atoi(argv[11]);
    int H2m2 = atoi(argv[12]);
    int H2p2 = atoi(argv[13]);
    int H3m2 = atoi(argv[14]);
    int H3p2 = atoi(argv[15]);
    int H1m3 = atoi(argv[16]);
    int H1p3 = atoi(argv[17]);
    int H2m3 = atoi(argv[18]);
    int H2p3 = atoi(argv[19]);
    int H3m3 = atoi(argv[20]);
    int H3p3 = atoi(argv[21]);

    halo_exchange_3D_generic_full::test(DIM1,
        DIM2,
        DIM3,
        H1m1,
        H1p1,
        H2m1,
        H2p1,
        H3m1,
        H3p1,
        H1m2,
        H1p2,
        H2m2,
        H2p2,
        H3m2,
        H3p2,
        H1m3,
        H1p3,
        H2m3,
        H2p3,
        H3m3,
        H3p3);

    MPI_Finalize();

}
