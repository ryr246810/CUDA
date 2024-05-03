#ifndef _NodeFlds_BufferWriter_HeaderFile
#define _NodeFlds_BufferWriter_HeaderFile

#include <NodeFlds_ConstBuffer.hxx>
#include <NodeFldsBase.hxx>

#include <vector>


#include <RgnWalker.hxx>

/**
 * BufferWriter works recursively through template
 * metaprogramming to write data into a buffer from
 * the values of a field in some region.
 */

class NodeFlds_BufferWriter : public NodeFlds_ConstBuffer
{
  friend class RgnWalker<NodeFlds_BufferWriter>;

public:
  NodeFlds_BufferWriter():NodeFlds_ConstBuffer(){
  }

  virtual ~NodeFlds_BufferWriter(){
  }


public:
  /*** Update by walking over the region */
  void UpdateVertices();
  void UpdateCells();


private:
  void UpdateVertex();
  void UpdateCell();


private:
  // Prevent use
  NodeFlds_BufferWriter(const NodeFlds_BufferWriter&);
  NodeFlds_BufferWriter& operator=(const NodeFlds_BufferWriter&);
};

#endif
