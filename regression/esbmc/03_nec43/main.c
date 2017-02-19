
/*
   An array with constant-time reset.
*/

#include <stdlib.h>

typedef int data_t;
typedef size_t idx_t;
typedef int bool_t;

//int __NONDET__();
int nondet_int();
//int __ESBMC_assume(int);
//int assert(int);

typedef struct {
  data_t resetVal;
  data_t *data;
  idx_t numData;
  idx_t maxNumData;
  idx_t *dataIdx;
  idx_t *dataWriteEvidence;
} buf_t;

buf_t *bufAlloc(size_t n) {
  int i;
  buf_t *b = (buf_t *)malloc(sizeof(buf_t));
  __ESBMC_assume(b);
  b->data = (data_t *)malloc(sizeof(data_t) * n);
  __ESBMC_assume(b->data);
  b->maxNumData = n;
  b->numData = 0;
  for (i=0; i<n; i++)
     b->dataWriteEvidence[i] = n;
  return b;
}

bool_t bufIdxWritten(const buf_t *buf_, idx_t idx_) {
  __ESBMC_assume(buf_ != NULL );
  __ESBMC_assume(0 <= idx_ );
  __ESBMC_assume(idx_ < buf_->maxNumData);
  return buf_->dataWriteEvidence[idx_] >= 0 &&
    buf_->dataWriteEvidence[idx_] < buf_->numData &&
    buf_->dataIdx[buf_->dataWriteEvidence[idx_]] == idx_;
}

data_t bufRead(const buf_t *buf_, idx_t idx_) {
  __ESBMC_assume(buf_ != NULL );
  __ESBMC_assume(0 <= idx_ );
  __ESBMC_assume( idx_ < buf_->maxNumData);
  return bufIdxWritten(buf_, idx_) ? buf_->data[buf_->dataWriteEvidence[idx_]] : buf_->resetVal;
}

void bufReset(buf_t *buf_, data_t resetVal_) {
  __ESBMC_assume(buf_ != NULL);
  buf_->resetVal = resetVal_;
  buf_->numData = 0;
}

void bufWrite(buf_t *buf_, idx_t idx_, data_t val_) {
   __ESBMC_assume(buf_!=NULL);
   __ESBMC_assume(0 <= idx_);
   __ESBMC_assume(idx_ < buf_->maxNumData);
   idx_t writeDataTo = buf_->dataWriteEvidence[idx_];
   if (!bufIdxWritten(buf_, idx_)) {
    assert(buf_->numData < buf_->maxNumData);
    buf_->dataIdx[buf_->numData] = idx_;
    buf_->dataWriteEvidence[idx_] = buf_->numData;
    writeDataTo = buf_->numData;
    buf_->numData++;
  }
  buf_->data[writeDataTo] = val_;
}

idx_t randomIdx(const buf_t *buf_) {
  __ESBMC_assume(buf_ != NULL);
  idx_t idx = nondet_int();
  __ESBMC_assume(0 <= idx);
  __ESBMC_assume(idx < buf_->maxNumData);
  return idx;
}

int main(int argc, char *argv[]) {
  const int numWrites = 4, numReads = 10, numBufs = 3, maxN = 20;
  int i,j;
  data_t datum;
  bool_t shouldReset;
  bool_t datumOut;
  
  buf_t **bufs = (buf_t **)malloc(numBufs * sizeof(buf_t *));
  __ESBMC_assume(bufs);
  for (i=0; i<numBufs; i++)
     bufs[i] = bufAlloc(maxN);
  
  for (i=0; i<numWrites; i++) {
     for (j=0; j<numBufs; j++)
        bufWrite(bufs[j], randomIdx(bufs[j]), (data_t)nondet_int());
     
  }
  
  for (i=0; i<numReads; i++) {
     for (j=0; j<numBufs; j++) {
        datum = (data_t)nondet_int();
        shouldReset = nondet_int();
        datumOut = (data_t)0;
        if (shouldReset)
           bufReset(bufs[j], datum);
        else
           datumOut = bufRead(bufs[j], randomIdx(bufs[j]));
        
        
     }
  }
  return 1;
}
