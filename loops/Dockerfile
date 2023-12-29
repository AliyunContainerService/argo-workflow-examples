FROM alpine
COPY print-pet.sh /tmp/print-pet.sh
COPY pets.input /tmp/pets.input
RUN chmod +x /tmp/print-pet.sh
ENTRYPOINT /tmp/print-pet.sh