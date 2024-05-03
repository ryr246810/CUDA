// --------------------------------------------------------------------
// File:	RgnWalker.h
// Purpose:	"Walk" through all the points on grid region and update
// --------------------------------------------------------------------

#ifndef _RgnWalker_HeaderFile
#define _RgnWalker_HeaderFile

#include <TxSlab2D.h>

/**
 * RgnWalker is used to do arbitrary dimensional updates.  It 
 * walks over a region using template recursion and calls the
 * UpdateVertex() or UpdateCell() method of the object that it is walking
 * around the region.  This object, the UPDATER, must have the two bump
 * methods and the UpdateVertex() or UpdateCell() method.  The latter is 
 * assumed to bump the updater in the last spatial dimension.
 */

template <class UPDATER> class RgnWalker
{
public:
  /** Walk through the region, rgn, and when going over the last dimension, 
      execute the update method of the updater object. **/
  static inline void walk_Vertex(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    walk_Vertex_0(rgn, updater);
  };


  /** Walk through the region, rgn, and when going over the last dimension, 
      execute the update method of the updater object. **/
  static inline void walk_XtndVertex(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    walk_XtndVertex_0(rgn, updater);
  };



  /** Walk through the region, rgn, and when going over the last dimension,
      execute the update method of the updater object. **/
  static inline void walk_Cell(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    walk_Cell_0(rgn, updater);
  };



private:
  static inline void walk_Vertex_0(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    // Do loop over next direction
    int lb=rgn.getLowerBound(0);
    int ub=rgn.getUpperBound(0);

    updater->bump(0, lb);

    for(int i=lb;i<=ub; ++i){
      walk_Vertex_1(rgn, updater);
      updater->bump(0);
    }

    // Take it back to the beginning
    updater->iBump(0, (ub+1) );
  };

  static inline void walk_Vertex_1(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    int lb=rgn.getLowerBound(1);
    int ub=rgn.getUpperBound(1);

    updater->bump(1, lb);

    for (int i=lb;i<=ub;++i){
      updater->UpdateVertex();
      updater->bump(1);
    }

    updater->iBump(1, (ub+1) );
  };


  static inline void walk_XtndVertex_0(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    // Do loop over next direction
    int lb=rgn.getLowerBound(0);
    int ub=rgn.getUpperBound(0);

    updater->bump(0, lb);

    for(int i=lb;i<ub; ++i){
      walk_XtndVertex_1(rgn, updater);
      updater->bump(0);
    }
    walk_XtndVertex_1(rgn, updater);

    // Take it back to the beginning
    updater->iBump(0, ub );
  };

  static inline void walk_XtndVertex_1(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    int lb=rgn.getLowerBound(1);
    int ub=rgn.getUpperBound(1);

    updater->bump(1, lb);

    for (int i=lb;i<ub;++i){
      updater->UpdateVertex();
      updater->bump(1);
    }
    updater->UpdateVertex();

    updater->iBump(1, ub );
  };


private:
  static inline void walk_Cell_0(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    // Do loop over next direction
    int lb=rgn.getLowerBound(0);
    int ub=rgn.getUpperBound(0);

    updater->bump(0, lb);

    for(int i=lb;i<ub; ++i){
      walk_Cell_1(rgn, updater);
      updater->bump(0);
    }

    // Take it back to the beginning
    updater->iBump(0, ub);
  };

  static inline void walk_Cell_1(const TxSlab2D<int>& rgn, UPDATER* updater)
  {
    int lb=rgn.getLowerBound(1);
    int ub=rgn.getUpperBound(1);

    updater->bump(1, lb);

    for (int i=lb;i<ub;++i){
      updater->UpdateCell();
      updater->bump(1);
    }

    updater->iBump(1, ub);
  };
};

#endif
