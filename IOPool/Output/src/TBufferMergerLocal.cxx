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

namespace ROOT {
namespace Experimental {

TBufferMergerLocal::TBufferMergerLocal(const char *name, Option_t *option, Int_t compress)
{
   // We cannot chain constructors or use in-place initialization here because
   // instantiating a TBufferMergerLocal should not alter gDirectory's state.
   TDirectory::TContext ctxt;
   Init(TFile::Open(name, option, /* title */ name, compress));
}

TBufferMergerLocal::TBufferMergerLocal(std::unique_ptr<TFile> output)
{
   Init(output.release());
}

void TBufferMergerLocal::Init(TFile *output)
{
   if (!output || !output->IsWritable() || output->IsZombie())
      Error("TBufferMergerLocal", "cannot write to output file");

   std::cout << "TBufferMergerLocal::Init\n";
   fMerger.OutputFile(std::unique_ptr<TFile>(output));
}

TBufferMergerLocal::~TBufferMergerLocal()
{
   for (auto f : fAttachedFiles)
      if (!f.expired()) Fatal("TBufferMergerLocal", " TBufferMergerLocalFiles must be destroyed before the server");

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

void TBufferMergerLocal::RegisterCallback(const std::function<void(void)> &f)
{
   fCallback = f;
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
   std::unique_lock<std::mutex> m(fMergeMutex, std::try_to_lock);
   if (m.owns_lock()) {
     {
        std::lock_guard<std::mutex> q(fQueueMutex);

        while (!fQueue.empty()) {
           std::unique_ptr<TBufferFile> buffer{fQueue.front()};
           fMerger.AddAdoptFile(
              new TMemFile(fMerger.GetOutputFileName(), buffer->Buffer(), buffer->BufferSize(), "READ"));
           fQueue.pop();
        }

        fBuffered = 0;
     }

     fMerger.PartialMerge();
     fMerger.Reset();

     if (fCallback)
        fCallback();
   } else {
     std::cout << "TBufferMergerLocal::Merge failed to acquire lock\n";
   }
}

} // namespace Experimental
} // namespace ROOT
