// @(#)root/io:$Id$
// Author: Philippe Canal, Witold Pokorski, and Guilherme Amadio

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferMerger_Local
#define ROOT_TBufferMerger_Local

#include "TFileMerger.h"
#include "TMemFile.h"

#include <functional>
#include <memory>
#include <mutex>
#include <queue>

namespace ROOT {
namespace Experimental {

class TBufferMergerFileLocal;

/**
 * \class TBufferMergerLocal TBufferMergerLocal.hxx
 * \ingroup IO
 *
 * TBufferMergerLocal is a class to facilitate writing data in
 * parallel from multiple threads, while writing to a single
 * output file. Its purpose is similar to TParallelMergingFile,
 * but instead of using processes that connect to a network
 * socket, TBufferMergerLocal uses threads that each write to a
 * TBufferMergerFileLocal, which in turn push data into a queue
 * managed by the TBufferMergerLocal.
 */

class TBufferMergerLocal {
public:
   /** Constructor
    * @param name Output file name
    * @param option Output file creation options
    * @param compression Output file compression level
    */
   TBufferMergerLocal(const char *name, Option_t *option = "RECREATE", Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose);

   /** Constructor
    * @param output Output \c TFile
    */
   TBufferMergerLocal(std::unique_ptr<TFile> output);

   /** Destructor */
   virtual ~TBufferMergerLocal();

   /** Returns a TBufferMergerFileLocal to which data can be written.
    *  At the end, all TBufferMergerFileLocals get merged into the output file.
    *  The user is responsible to "cd" into the file to associate objects
    *  such as histograms or trees to it.
    *
    *  After the creation of this file, the user must reset the kMustCleanup
    *  bit on any objects attached to it and take care of their deletion, as
    *  there is a possibility that a race condition will happen that causes
    *  a crash if ROOT manages these objects.
    */
   std::shared_ptr<TBufferMergerFileLocal> GetFile();

   /** Returns the number of buffers currently in the queue. */
   size_t GetQueueSize() const;

   /** Returns the current value of the auto save setting in bytes (default = 0). */
   size_t GetAutoSave() const;

   /** By default, TBufferMergerLocal will call TFileMerger::PartialMerge() for each
    *  buffer pushed onto its merge queue. This function lets the user change
    *  this behaviour by telling TBufferMergerLocal to accumulate at least @param size
    *  bytes in memory before performing a partial merge and flushing to disk.
    *  This can be useful to avoid an excessive amount of work to happen in the
    *  output thread, as the number of TTree headers (which require compression)
    *  written to disk can be reduced.
    */
   void SetAutoSave(size_t size);

   friend class TBufferMergerFileLocal;

private:
   /** TBufferMergerLocal has no default constructor */
   TBufferMergerLocal();

   /** TBufferMergerLocal has no copy constructor */
   TBufferMergerLocal(const TBufferMergerLocal &);

   /** TBufferMergerLocal has no copy operator */
   TBufferMergerLocal &operator=(const TBufferMergerLocal &);

   void Init(std::unique_ptr<TFile>);

   void Merge();
   void Push(TBufferFile *buffer);

   size_t fAutoSave{0};                                          //< AutoSave only every fAutoSave bytes
   size_t fBuffered{0};                                          //< Number of bytes currently buffered
   TFileMerger fMerger{false, false};                            //< TFileMerger used to merge all buffers
   std::mutex fMergeMutex;                                       //< Mutex used to lock fMerger
   std::mutex fQueueMutex;                                       //< Mutex used to lock fQueue
   std::queue<TBufferFile *> fQueue;                             //< Queue to which data is pushed and merged
   std::vector<std::weak_ptr<TBufferMergerFileLocal>> fAttachedFiles; //< Attached files
};

/**
 * \class TBufferMergerLocal TBufferMergerLocal.hxx
 * \ingroup IO
 *
 * A TBufferMergerFileLocal is similar to a TMemFile, but when data
 * is written to it, it is appended to the TBufferMergerLocal queue.
 * The TBufferMergerLocal merges all data into the output file on disk.
 */

class TBufferMergerFileLocal : public TMemFile {
private:
   TBufferMergerLocal &fMerger; //< TBufferMergerLocal this file is attached to

   /** Constructor. Can only be called by TBufferMergerLocal.
    * @param m Merger this file is attached to. */
   TBufferMergerFileLocal(TBufferMergerLocal &m);

   /** TBufferMergerFileLocal has no default constructor. */
   TBufferMergerFileLocal();

   /** TBufferMergerFileLocal has no copy constructor. */
   TBufferMergerFileLocal(const TBufferMergerFileLocal &);

   /** TBufferMergerFileLocal has no copy operator */
   TBufferMergerFileLocal &operator=(const TBufferMergerFileLocal &);

   friend class TBufferMergerLocal;

public:
   /** Destructor */
   ~TBufferMergerFileLocal();

   using TMemFile::Write;

   /** Write data into a TBufferFile and append it to TBufferMergerLocal.
    * @param name Name
    * @param opt  Options
    * @param bufsize Buffer size
    * This function must be called before the TBufferMergerFileLocal gets destroyed,
    * or no data is appended to the TBufferMergerLocal.
    */
   virtual Int_t Write(const char *name = nullptr, Int_t opt = 0, Int_t bufsize = 0) override;

   //ClassDefOverride(TBufferMergerFileLocal, 0);
};

} // namespace Experimental
} // namespace ROOT

#endif
