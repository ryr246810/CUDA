//-----------------------------------------------------------------------------
//
// File:	PrRefCount.h:
//
// Purpose:
//
// Version:	$Id: PrRefCount.h 121 2007-11-21 21:07:25Z swsides $
//
// Copyright 1998, 1999 Tech-X Corporation
//
//-----------------------------------------------------------------------------

#ifndef PR_REF_COUNT_H
#define PR_REF_COUNT_H

// Include definition of size_t
#include <vector>
#include "TxBase.h"
/**
 *    --  Base class for reference counting.
 *
 *  Review of reference counting technique:
 *    The envelope class determines whether the copy constructor is used.
 *    If the class is changing the internal data, then make a new copy.
 *
 *  TxRefCount is the envelope class -- it presents the interface to the world.
 *  PrRefCount is the  letter  class -- it does the work and keeps track of references.
 *
 *  Copyright 1996, 1997, 1998 by Tech-X Corporation
 *
 *  @author  John R. Cary
 *
 *  @version $Id: PrRefCount.h 121 2007-11-21 21:07:25Z swsides $
 */
class TXBASE_API PrRefCount 
{

  friend class TxRefCount;

  public:

/**
 * Create a contained object
 */
    PrRefCount() { 
      refCount = 1;
    }

/**
 * Copy constructor
 */
    PrRefCount(const PrRefCount& prc) { 
      refCount = 1;
    }

/**
 * destroy the contained object
 */
    virtual ~PrRefCount() { 
    }

/**
 * get number of references to this object
 *
 * @return the number of references
 */
    int getRefCount() const { return refCount;} 

/**
 * decrease the count 
 */
    void decrementCount() { refCount--;}

/**
 *  increase count 
 */
    void incrementCount() { refCount++;}

  protected:

/**
 * The number of references to this object.
 */
    int refCount;

  private:
/**
 * Assignment operator
 */
    PrRefCount& operator=(const PrRefCount& prc);

};

#endif	 // PR_REF_COUNT
