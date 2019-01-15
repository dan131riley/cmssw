// @(#)root/io:$Id$
// Author: Philippe Canal, Witold Pokorski, and Guilherme Amadio

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBufferMergerLocal.hxx"

#include "TBufferFile.h"
#include "TError.h"
#include "TROOT.h"
#include "TVirtualMutex.h"

#include <iostream>
#include <utility>

namespace ROOT {
namespace Experimental {

TBufferMergerLocal::TBufferMergerLocal(const char *name, Option_t *option, Int_t compress)
{
   // We cannot chain constructors or use in-place initialization here because
   // instantiating a TBufferMergerLocal should not alter gDirectory's state.
   TDirectory::TContext ctxt;
   Init(std::unique_ptr<TFile>(TFile::Open(name, option, /* title */ name, compress)));
}

TBufferMergerLocal::TBufferMergerLocal(std::unique_ptr<TFile> output)
{
   Init(std::move(output));
}

void TBufferMergerLocal::Init(std::unique_ptr<TFile> output)
{
   if (!output || !output->IsWritable() || output->IsZombie())
      Error("TBufferMergerLocal", "cannot write to output file");

   fMerger.OutputFile(std::move(output));
}

TBufferMergerLocal::~TBufferMergerLocal()
{
   for (const auto &f : fAttachedFiles)
      if (!f.expired()) Fatal("TBufferMergerLocal", " TBufferMergerFileLocals must be destroyed before the server");

   if (!fQueue.empty())
      Merge();
}

std::shared_ptr<TBufferMergerFileLocal> TBufferMergerLocal::GetFile()
{
   R__LOCKGUARD(gROOTMutex);
   std::shared_ptr<TBufferMergerFileLocal> f(new TBufferMergerFileLocal(*this));
   gROOT->GetListOfFiles()->Remove(f.get());
   fAttachedFiles.push_back(f);
   return f;
}

size_t TBufferMergerLocal::GetQueueSize() const
{
   return fQueue.size();
}

void TBufferMergerLocal::Push(TBufferFile *buffer)
{
   {
      std::lock_guard<std::mutex> lock(fQueueMutex);
      fBuffered += buffer->BufferSize();
      fQueue.push(buffer);
   }

   if (fBuffered > fAutoSave)
      Merge();
}

size_t TBufferMergerLocal::GetAutoSave() const
{
   return fAutoSave;
}

void TBufferMergerLocal::SetAutoSave(size_t size)
{
   fAutoSave = size;
}

void TBufferMergerLocal::Merge()
{
   if (fMergeMutex.try_lock()) {
      std::queue<TBufferFile *> queue;
      {
         std::lock_guard<std::mutex> q(fQueueMutex);
         std::swap(queue, fQueue);
         fBuffered = 0;
      }

      while (!queue.empty()) {
         std::unique_ptr<TBufferFile> buffer{queue.front()};
         fMerger.AddAdoptFile(
            new TMemFile(fMerger.GetOutputFileName(), buffer->Buffer(), buffer->BufferSize(), "READ"));
         queue.pop();
      }

      fMerger.PartialMerge(TFileMerger::kAll | TFileMerger::kIncremental | TFileMerger::kKeepCompression);
      fMerger.Reset();
      fMergeMutex.unlock();
   } else {
      std::cout << "TBufferMergerLocal::Merge failed to acquire lock\n";
   }
}

} // namespace Experimental
} // namespace ROOT
