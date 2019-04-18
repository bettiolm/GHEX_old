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

#include <gridtools/meta/utility.hpp>
#include <prototype/generic_interfaces.hpp>
#include <prototype/range_loops.hpp>
#include <prototype/regular_grid_descriptors.hpp>
#include <prototype/triplet.hpp>

namespace gt = gridtools;

using id_type = std::array<int, 3>;


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


int main(int argc, char** argv) {

    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int grid_sizes[3] = {0,0,0};

    MPI_Dims_create(world_size, 3, grid_sizes);

    show(grid_sizes[0]);
    show(grid_sizes[1]);
    show(grid_sizes[2]);

    std::stringstream ss;
    ss << rank;
    std::string filename = "out" + ss.str() + ".txt";
    std::cout << filename << std::endl;
    std::ofstream file(filename.c_str());

    // local ids: there is 1 domain per node, and it is identified by a triplet of indices <i,j,k>
    std::list<id_type> local_ids{ {rank%(grid_sizes[0]), (rank/(grid_sizes[0])%grid_sizes[1]), rank/((grid_sizes[0])*(grid_sizes[1]))} };

    file << "Local ids\n";
    std::for_each(local_ids.begin(), local_ids.end(), [&file] (id_type const& x) { file << x << " ";});
    file << "\n";

    auto neighbor_generator = [grid_sizes](id_type id) -> std::array<std::pair<id_type, dir_type>, 6> { // this lambda returns a sequence of neighbors of a local_id
        int i = id[0];
        int j = id[1];
        int k = id[2];
        return {
            std::make_pair(id_type{mod(i-1, grid_sizes[0]), j, k}, std::array<int, 3>{-1,0,0}),
            std::make_pair(id_type{mod(i+1, grid_sizes[0]), j, k}, std::array<int, 3>{1,0,0}),
            std::make_pair(id_type{i, mod(j-1, grid_sizes[1]), k}, std::array<int, 3>{0,-1,0}),
            std::make_pair(id_type{i, mod(j+1, grid_sizes[1]), k}, std::array<int, 3>{0,1,0}),
            std::make_pair(id_type{i, j, mod(k-1, grid_sizes[2])}, std::array<int, 3>{0,0,-1}),
            std::make_pair(id_type{i, j, mod(k+1, grid_sizes[2])}, std::array<int, 3>{0,0,1}),
        };
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

    // Generating the PG with the sequence of local IDs and a function to gather neighbors
    generic_pg<id_type, dir_type> pg(local_ids, neighbor_generator, file);

    pg.show_topology(file);
    file.flush();

    /* ===== Data preparation, with halos ===== */

    typedef double T1;
    typedef gridtools::layout_map<0, 1, 2> layoutmap;
    typedef data_descriptor_t<3, array<triple_t<USE_DOUBLE, T1>, layoutmap>, triple_t<USE_DOUBLE, T1>> data_dsc_type;

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
//                a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>(
//                    ii - H1m + (DIM1)*coords[0], jj - H2m + (DIM2)*coords[1], kk - H3m + (DIM3)*coords[2]);
                a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>(rank, rank, rank);
            }
        }
    }

    data_dsc_type data_dsc{
        a, (DIM1 + H1m + H1p), (DIM2 + H2m + H2p), (DIM3 + H3m + H3p)
    };

    // data_dsc.show_data(file);

    std::array<gt::halo_sizes, 3> halos = {gt::halo_sizes{H1m, H1p}, gt::halo_sizes{H2m, H2p}, gt::halo_sizes{H3m, H3p}};

    file.close();

    /* ===== End data preparation, with halos ===== */

    gt::regular_grid_descriptor< 3 /* number of partitioned dimensions */ > grid(halos);

    // Iteration spaces describe there the data to send and data to
    // receive. An iteration space for a communication object is a
    // function that takes the local id and the remote id (should it
    // take just the remote id, since the local is just the one to
    // which it is associated with?), and return the region to
    // pack/unpack. For regula grids this will take the iteration
    // ranges prototyped in some file here.
    auto iteration_spaces_send = [&data_dsc, &grid](id_type local, id_type remote, dir_type direction) {
        return grid.inner_iteration_space< gt::partitioned<0, 1, 2>>(data_dsc, direction);
    };

    auto iteration_spaces_recv = [&data_dsc, &grid](id_type local, id_type remote, dir_type direction) {
        return grid.outer_iteration_space< gt::partitioned<0, 1, 2>>(data_dsc, direction);
    };

    // constructing the communication object with the id associated to it and the topology information (processing grid)
    using co_type = generic_co<generic_pg<id_type, dir_type>, decltype(iteration_spaces_send), decltype(iteration_spaces_recv)>;
    // We need a CO object for each sub-domain in each rank
    std::vector<co_type> co;
    for (auto id : local_ids) {
        co.push_back(co_type{id, pg, iteration_spaces_send, iteration_spaces_recv});
    }

    // launching the computations
    auto itc = co.begin();
    for (auto it = local_ids.begin(); it != local_ids.end(); ++it, ++itc) {
        std::stringstream ss;
        ss << rank;
        std::string filename = "tout" + ss.str() + ".txt";
        std::cout << filename << std::endl;
        std::ofstream tfile(filename.c_str());
        tfile << "\nFILE for " << *it << "\n";
        data_dsc.show_data(tfile);
        tfile << "================================\n";

        auto hdl = (*itc).exchange<data_dsc_type, triple_t<USE_DOUBLE, T1>>(data_dsc, tfile);
        hdl.wait();

        data_dsc.show_data(tfile);
        tfile << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n";
        tfile.flush();
        tfile.close();
    }

    MPI_Finalize();

}
