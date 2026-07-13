#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define min(x, y) (((x) < (y))? (x) : (y))

void fill_edgeToRow_cuda(int* edgeToRow, int *nodePointer, int num_nodes);
void fill_window_cuda(int* edgeToColumn, int* blockPartition, int* nodePointer,
                      int* edgeList, int blockSize_h, int blockSize_w, int num_nodes);

std::vector<torch::Tensor> spmm_forward_cuda(
      torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
  );

std::vector<torch::Tensor> spmm_forward_cuda_with_value(
      torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition,
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input,
    torch::Tensor values
  );

std::vector<torch::Tensor> spmmAGNN_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor edgeAttention,        // *edge attention [n_head, n_e]
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow,
              int num_nodes,
              int num_edges,
              int embedding_dim,
    torch::Tensor input
    ); 

std::vector<torch::Tensor> sddmm_forward_cuda(
    torch::Tensor nodePointer,
    torch::Tensor edgeList,			    // edge list.
    torch::Tensor blockPartition,		// number of TC_blocks (16x8) in each row_window.
    torch::Tensor edgeToColumn, 		// eid -> col within each row_window.
    torch::Tensor edgeToRow, 			  // eid -> col within each row_window.
              int num_nodes,
              int num_edges,
              int embedding_dim,	    // embedding dimension.
	  torch::Tensor input				        // input feature matrix.
); 

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//////////////////////////////////////////
//
// SPMM Foward Pass (GCN, GraphSAGE)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_cuda(nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            input);
}

std::vector<torch::Tensor> spmm_forward_with_value(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor blockPartition,
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_CUDA(values);
  CHECK_CONTIGUOUS(values);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "values must have dtype float32");
  TORCH_CHECK(values.dim() == 1, "values must be one-dimensional");
  TORCH_CHECK(values.size(0) == edgeList.size(0),
              "values length must equal CSR nnz");

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

  return spmm_forward_cuda_with_value(nodePointer, edgeList,
                                      blockPartition, edgeToColumn, edgeToRow,
                                      num_nodes, num_edges, embedding_dim,
                                      input, values);
}

////////////////////////////////////////////
//
// SPMM Foward Pass (AGNN, AGNN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_AGNN(
    torch::Tensor input,
    torch::Tensor nodePointer,
    torch::Tensor edgeList,
    torch::Tensor edgeAttention,
    torch::Tensor blockPartition, 
    torch::Tensor edgeToColumn,
    torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(edgeAttention);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);
  
  return spmmAGNN_forward_cuda(nodePointer, edgeList, edgeAttention,
                              blockPartition, edgeToColumn, edgeToRow, 
                              num_nodes, num_edges, embedding_dim,
                              input);
}


////////////////////////////////////////////
//
// SDDMM Foward Pass
//
////////////////////////////////////////////
std::vector<torch::Tensor> sddmm_forward(
  torch::Tensor input,				
  torch::Tensor nodePointer,
  torch::Tensor edgeList,			    
	torch::Tensor blockPartition,		
	torch::Tensor edgeToColumn, 		
	torch::Tensor edgeToRow
) {
  CHECK_INPUT(input);
  CHECK_INPUT(nodePointer);
  CHECK_INPUT(edgeList);
  CHECK_INPUT(blockPartition);
  CHECK_INPUT(edgeToColumn);
  CHECK_INPUT(edgeToRow);

  int num_nodes = nodePointer.size(0) - 1;
  int num_edges = edgeList.size(0);
  int embedding_dim = input.size(1);

//   printf("at sddmm_forward\n");
  return sddmm_forward_cuda(nodePointer, edgeList, 
                            blockPartition, edgeToColumn, edgeToRow, 
                            num_nodes, num_edges, embedding_dim,
                            input);
}


// condense an sorted array with duplication: [1,2,2,3,4,5,5]
// after condense, it becomes: [1,2,3,4,5].
// Also, mapping the origin value to the corresponding new location in the new array.
// 1->[0], 2->[1], 3->[2], 4->[3], 5->[4]. 
std::map<unsigned, unsigned> inplace_deduplication(unsigned* array, unsigned length){
    std::map<unsigned, unsigned> nb2col;
    if (length == 0) {
        return nb2col;
    }

    unsigned loc = 0, cur = 1;
    nb2col[array[0]] = 0;
    while (cur < length){
        if(array[cur] != array[cur - 1]){
            loc++;
            array[loc] = array[cur];
            nb2col[array[cur]] = loc;       // mapping from eid to TC_block column index.[]
        }
        cur++;
    }
    return nb2col;
}

void preprocess(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                ){

    TORCH_CHECK(num_nodes >= 0, "num_nodes must be non-negative");
    TORCH_CHECK(blockSize_h > 0, "blockSize_h must be positive");
    TORCH_CHECK(blockSize_w > 0, "blockSize_w must be positive");
    TORCH_CHECK(!edgeList_tensor.is_cuda(), "edgeList must be a CPU tensor");
    TORCH_CHECK(!nodePointer_tensor.is_cuda(), "nodePointer must be a CPU tensor");
    TORCH_CHECK(!blockPartition_tensor.is_cuda(), "blockPartition must be a CPU tensor");
    TORCH_CHECK(!edgeToColumn_tensor.is_cuda(), "edgeToColumn must be a CPU tensor");
    TORCH_CHECK(!edgeToRow_tensor.is_cuda(), "edgeToRow must be a CPU tensor");
    TORCH_CHECK(edgeList_tensor.is_contiguous(), "edgeList must be contiguous");
    TORCH_CHECK(nodePointer_tensor.is_contiguous(), "nodePointer must be contiguous");
    TORCH_CHECK(blockPartition_tensor.is_contiguous(), "blockPartition must be contiguous");
    TORCH_CHECK(edgeToColumn_tensor.is_contiguous(), "edgeToColumn must be contiguous");
    TORCH_CHECK(edgeToRow_tensor.is_contiguous(), "edgeToRow must be contiguous");
    TORCH_CHECK(edgeList_tensor.dim() == 1, "edgeList must be one-dimensional");
    TORCH_CHECK(nodePointer_tensor.dim() == 1, "nodePointer must be one-dimensional");
    TORCH_CHECK(blockPartition_tensor.dim() == 1, "blockPartition must be one-dimensional");
    TORCH_CHECK(edgeToColumn_tensor.dim() == 1, "edgeToColumn must be one-dimensional");
    TORCH_CHECK(edgeToRow_tensor.dim() == 1, "edgeToRow must be one-dimensional");
    TORCH_CHECK(edgeList_tensor.scalar_type() == torch::kInt32, "edgeList must have dtype int32");
    TORCH_CHECK(nodePointer_tensor.scalar_type() == torch::kInt32, "nodePointer must have dtype int32");
    TORCH_CHECK(blockPartition_tensor.scalar_type() == torch::kInt32, "blockPartition must have dtype int32");
    TORCH_CHECK(edgeToColumn_tensor.scalar_type() == torch::kInt32, "edgeToColumn must have dtype int32");
    TORCH_CHECK(edgeToRow_tensor.scalar_type() == torch::kInt32, "edgeToRow must have dtype int32");
    TORCH_CHECK(nodePointer_tensor.size(0) >= static_cast<int64_t>(num_nodes) + 1,
                "nodePointer must contain at least num_nodes + 1 entries");

    // input tensors.
    auto edgeList = edgeList_tensor.accessor<int, 1>();
    auto nodePointer = nodePointer_tensor.accessor<int, 1>();

    TORCH_CHECK(nodePointer[0] == 0, "nodePointer[0] must be zero");
    for (int nid = 0; nid < num_nodes; ++nid) {
        TORCH_CHECK(nodePointer[nid] >= 0 && nodePointer[nid] <= nodePointer[nid + 1],
                    "nodePointer must be non-negative and non-decreasing");
    }
    const int64_t num_edges = nodePointer[num_nodes];
    TORCH_CHECK(num_edges <= edgeList_tensor.size(0),
                "nodePointer exceeds edgeList capacity");
    const int64_t num_windows =
        (static_cast<int64_t>(num_nodes) + blockSize_h - 1) / blockSize_h;
    TORCH_CHECK(blockPartition_tensor.size(0) >= num_windows,
                "blockPartition is smaller than the number of row windows");
    TORCH_CHECK(edgeToColumn_tensor.size(0) >= num_edges,
                "edgeToColumn is smaller than the CSR edge count");
    TORCH_CHECK(edgeToRow_tensor.size(0) >= num_edges,
                "edgeToRow is smaller than the CSR edge count");

    // output tensors.
    auto blockPartition = blockPartition_tensor.accessor<int, 1>();
    auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
    auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();

    unsigned block_counter = 0;

    #pragma omp parallel for 
    for (unsigned nid = 0; nid < num_nodes; nid++){
        for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
            edgeToRow[eid] = nid;
    }

    #pragma omp parallel for reduction(+:block_counter)
    for (unsigned windowId = 0; windowId < num_windows; ++windowId){
        unsigned iter = windowId * blockSize_h;
        unsigned block_start = nodePointer[iter];
        unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
        unsigned num_window_edges = block_end - block_start;

        if (num_window_edges == 0) {
            blockPartition[windowId] = 0;
            continue;
        }

        unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
        memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

        // Step-1: Sort the neighbor id array of a row window.
        thrust::sort(neighbor_window, neighbor_window + num_window_edges);

        // Step-2: Deduplication of the edge id array.
        // printf("Before dedupblication: %d\n", num_window_edges);
        std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);

        // generate blockPartition --> number of TC_blcok in each row window.
        blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
        block_counter += blockPartition[windowId];

        // scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
        for (unsigned e_index = block_start; e_index < block_end; e_index++){
            unsigned eid = edgeList[e_index];
            edgeToColumn[e_index] = clean_edges2col[eid];
        }
        free(neighbor_window);
    }
    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
}


void preprocess_gpu(torch::Tensor edgeList_tensor, 
                torch::Tensor nodePointer_tensor, 
                int num_nodes, 
                int blockSize_h,
                int blockSize_w,
                torch::Tensor blockPartition_tensor, 
                torch::Tensor edgeToColumn_tensor,
                torch::Tensor edgeToRow_tensor
                )
{

    TORCH_CHECK(num_nodes >= 0, "num_nodes must be non-negative");
    TORCH_CHECK(blockSize_h > 0, "blockSize_h must be positive");
    TORCH_CHECK(blockSize_w > 0, "blockSize_w must be positive");
    CHECK_INPUT(edgeList_tensor);
    CHECK_INPUT(nodePointer_tensor);
    CHECK_INPUT(blockPartition_tensor);
    CHECK_INPUT(edgeToColumn_tensor);
    CHECK_INPUT(edgeToRow_tensor);
    TORCH_CHECK(edgeList_tensor.scalar_type() == torch::kInt32, "edgeList must have dtype int32");
    TORCH_CHECK(nodePointer_tensor.scalar_type() == torch::kInt32, "nodePointer must have dtype int32");
    TORCH_CHECK(blockPartition_tensor.scalar_type() == torch::kInt32, "blockPartition must have dtype int32");
    TORCH_CHECK(edgeToColumn_tensor.scalar_type() == torch::kInt32, "edgeToColumn must have dtype int32");
    TORCH_CHECK(edgeToRow_tensor.scalar_type() == torch::kInt32, "edgeToRow must have dtype int32");
    TORCH_CHECK(nodePointer_tensor.size(0) >= static_cast<int64_t>(num_nodes) + 1,
                "nodePointer must contain at least num_nodes + 1 entries");
    const int64_t num_windows =
        (static_cast<int64_t>(num_nodes) + blockSize_h - 1) / blockSize_h;
    TORCH_CHECK(blockPartition_tensor.size(0) >= num_windows,
                "blockPartition is smaller than the number of row windows");
    TORCH_CHECK(edgeToColumn_tensor.size(0) >= edgeList_tensor.size(0),
                "edgeToColumn is smaller than edgeList");
    TORCH_CHECK(edgeToRow_tensor.size(0) >= edgeList_tensor.size(0),
                "edgeToRow is smaller than edgeList");

    // input tensors.
    auto edgeList = edgeList_tensor.data<int>();
    auto nodePointer = nodePointer_tensor.data<int>();

    // output tensors.
    auto blockPartition = blockPartition_tensor.data<int>();
    auto edgeToColumn = edgeToColumn_tensor.data<int>();
    auto edgeToRow = edgeToRow_tensor.data<int>();

    unsigned block_counter = 0;
    
    fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
    fill_window_cuda(edgeToColumn, blockPartition, nodePointer, edgeList,
                                blockSize_h, blockSize_w, num_nodes);

    printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess", &preprocess, "Preprocess Step (CPU)");
  m.def("preprocess_gpu", &preprocess_gpu, "Preprocess Step (CUDA)");

  // forward computation
  m.def("forward", &spmm_forward, "TC-GNN SPMM forward (CUDA)");
  m.def("forward_with_value", &spmm_forward_with_value,
        "TC-GNN valued SPMM forward (CUDA)");
  m.def("forward_ef", &sddmm_forward, "TC-GNN SDDMM forward (CUDA)");
  m.def("forward_AGNN", &spmm_forward_AGNN, "TC-GNN SPMM (AGNN) forward (CUDA)");

  // backward
  m.def("backward", &spmm_forward, "TC-GNN SPMM backward (CUDA)");
  m.def("backward_ef", &sddmm_forward, "TC-GNN SDDMM backward_ef (CUDA)");
}
