//-----------------------------------------------------------------------------
//
// File:	TxRefCount.h
//
// Purpose:	Container of reference counted object.  Handles bookkeeping.
//
// Version:	$Id: TxRefCount.h 52 2006-07-26 21:20:47Z yew $
//
// Copyright 1997, 1998, 1999 Tech-X Corporation
//
//-----------------------------------------------------------------------------

#ifndef TX_REF_COUNT_H
#define TX_REF_COUNT_H

// txstd includes
#include "PrRefCount.h"
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
 *  @version $Id: TxRefCount.h 52 2006-07-26 21:20:47Z yew $
 */
class TXBASE_API TxRefCount 
{
 public:
  
  // Constructors
  /**
   * Construct from a PrRefCount: primarily of internal and derived use
   * when making unique.
   */
  TxRefCount(PrRefCount* p) {
    prc = p;
  }
  
  /**
   * Default constructor
   TxRefCount() {
   }
  */
  
  /**
   * Copy constructor: bump number of references.
   */
  TxRefCount(const TxRefCount& trc) { 
    prc = trc.prc;
    prc->refCount++;
  }
  
  /**
   * Destructor
   */
  virtual ~TxRefCount() { 
    prc->refCount--;
    if (!prc->refCount) delete prc;
  }
  
  /**
   * Get the number of references to the internal data.
   */
  int getRefCount() const { return prc->getRefCount();}
  
  /**
   * Assignment: usual ref count scheme.
   */
  TxRefCount& operator=(const TxRefCount& trc) {
    if ( this == &trc ) return *this;
    prc->refCount--;
    if (!prc->refCount) delete prc;
    prc = trc.prc;
    prc->refCount++;
    return *this;
  }
  
 protected:
  
  /**
   * pointer to the PrRefCount object (the letter) which does all the work
   */
  PrRefCount*	prc;
};

#endif	 // TX_REF_COUNT_H
