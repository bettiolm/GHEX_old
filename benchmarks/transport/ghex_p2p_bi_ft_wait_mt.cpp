#include <iostream>
#include <vector>
#include <atomic>

#include <ghex/common/timer.hpp>
#include "utils.hpp"

namespace ghex = gridtools::ghex;

#ifdef USE_MPI
/* MPI backend */
#ifdef USE_OPENMP
#include <ghex/threads/atomic/primitives.hpp>
using threading    = ghex::threads::atomic::primitives;
#else
#include <ghex/threads/none/primitives.hpp>
using threading    = ghex::threads::none::primitives;
#endif
#include <ghex/transport_layer/mpi/context.hpp>
using transport    = ghex::tl::mpi_tag;
#else
/* UCX backend */
#ifdef USE_OPENMP
#include <ghex/threads/omp/primitives.hpp>
using threading    = ghex::threads::omp::primitives;
#else
#include <ghex/threads/none/primitives.hpp>
using threading    = ghex::threads::none::primitives;
#endif
#include <ghex/transport_layer/ucx/address_db_mpi.hpp>
#include <ghex/transport_layer/ucx/context.hpp>
using db_type      = ghex::tl::ucx::address_db_mpi;
using transport    = ghex::tl::ucx_tag;
#endif /* USE_MPI */

#include <ghex/transport_layer/message_buffer.hpp>

using context_type = ghex::tl::context<transport, threading>;
using communicator_type = typename context_type::communicator_type;
using future_type = typename communicator_type::future<void>;


using MsgType = gridtools::ghex::tl::message_buffer<>;


int main(int argc, char *argv[])
{
    int niter, buff_size;
    int inflight;
    int mode;
    gridtools::ghex::timer timer, ttimer;

    if(argc != 4)
    {
        std::cerr << "Usage: bench [niter] [msg_size] [inflight]" << "\n";
        std::terminate();
    }
    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);

    int num_threads = 1;
#ifdef USE_OPENMP
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
        std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
        std::terminate();
    }
#pragma omp parallel
    {
#pragma omp master
        num_threads = omp_get_num_threads();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif

    {
        auto context_ptr = ghex::tl::context_factory<transport,threading>::create(num_threads, MPI_COMM_WORLD);
        auto& context = *context_ptr;

#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
            auto token             = context.get_token();
            auto comm              = context.get_communicator(token);
            const auto rank        = comm.rank();
            const auto size        = comm.size();
            const auto thread_id   = token.id();
            const auto num_threads = context.thread_primitives().size();
            const auto peer_rank   = (rank+1)%2;

            bool using_mt = false;
#ifdef USE_OPENMP
            using_mt = true;
#endif

            if (thread_id==0 && rank==0)
            {
                std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
            };

            std::vector<MsgType> smsgs(inflight);
            std::vector<MsgType> rmsgs(inflight);
            std::vector<future_type> sreqs(inflight);
            std::vector<future_type> rreqs(inflight);
            for(int j=0; j<inflight; j++)
            {
                smsgs[j].resize(buff_size);
                rmsgs[j].resize(buff_size);
                make_zero(smsgs[j]);
                make_zero(rmsgs[j]);
            }

            comm.barrier();

            if(thread_id == 0)
            {
                timer.tic();
                ttimer.tic();
                if(rank == 1)
                    std::cout << "number of threads: " << num_threads << ", multi-threaded: " << using_mt << "\n";
            }

            int dbg = 0;
            int sent = 0, received = 0;
            int last_received = 0;
            int last_sent = 0;
            while(sent < niter || received < niter){

                if(thread_id == 0 && dbg >= (niter/10)) {
                    dbg = 0;
                    std::cout << rank << " total bwdt MB/s:      "
                              << ((double)(received-last_received + sent-last_sent)*size*buff_size/2)/timer.toc()
                              << "\n";
                    timer.tic();
                    last_received = received;
                    last_sent = sent;
                }

                /* submit comm */
                for(int j=0; j<inflight; j++){

                    dbg += num_threads;
                    sent += num_threads;
                    received += num_threads;

                    rreqs[j] = comm.recv(rmsgs[j], peer_rank, thread_id*inflight + j);
                    sreqs[j] = comm.send(smsgs[j], peer_rank, thread_id*inflight + j);
                }

                /* wait for all */
                for(int j=0; j<inflight; j++){
                    sreqs[j].wait();
                    rreqs[j].wait();
                }
            }

            comm.barrier();
            if(thread_id == 0 && rank == 0){
                const auto t = ttimer.toc();
                std::cout << "time:       " << t/1000000 << "s\n";
                std::cout << "final MB/s: " << ((double)niter*size*buff_size)/t << "\n";
            }
        }

        // tail loops - not needed in wait benchmarks
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
